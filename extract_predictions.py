import torchvision.transforms as ttf
from src.factory import *
from tqdm import tqdm
import sys
import torch
import os
import argparse
from src.validate import validate
from scipy.spatial.transform import Rotation as R
import numpy as np
import faiss

msls_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'val': ["cph", "sf"],
    'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
}


class TestParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--dataset', required=True, default='MSLS',
                                 help='Name of the dataset [MSLS|7Scenes|TB_Places]')

        self.parser.add_argument('--root_dir', required=True, help='Root directory of the dataset')
        self.parser.add_argument('--subset', required=False, default='val', help='For MSLS. Subset to test')

        self.parser.add_argument('--query_idx_file', type=str, required=False, help='Query idx file, .json')
        self.parser.add_argument('--map_idx_file', type=str, required=False, help='Map idx file, .json')

        self.parser.add_argument('--model_file', type=str, required=True, help='Model file, .pth')
        self.parser.add_argument('--backbone', type=str, default='resnet50',
                                 help='architectures: [vgg16, resnet18, resnet34, resnet50, resnet152, densenet161]')
        self.parser.add_argument('--pool', type=str, required=True, help='pool type|avg,max,GeM', default='GeM')
        self.parser.add_argument('--f_length', type=int, default=2048, help='feature length')
        self.parser.add_argument('--image_size', type=str, default="480,640", help='Input size, separated by commas')
        self.parser.add_argument('--norm', type=str, default="L2", help='Normalization descriptors')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size')

    def parse(self):
        self.opt = self.parser.parse_args()


def extract_features(dl, net, f_length, feats_file):
    if not os.path.exists(feats_file):
        feats = np.zeros((len(dl.dataset), f_length))
        for i, batch in tqdm(enumerate(dl), desc="Extracting features"):
            if torch.cuda.is_available():
                x = net.forward(batch.cuda())
            else:
                x = net.forward(batch)
            feats[i * dl.batch_size:i * dl.batch_size + dl.batch_size] = x.cpu().detach().squeeze(0)

        np.save(feats_file, feats)
    else:
        print(feats_file, "already exists. Skipping.")


def extract_features_msls(subset, root_dir, net, f_length, image_t, savename, results_dir,
                          batch_size, k, cls_token=False):
    cities = default_cities[subset]

    result_file = results_dir + "/" + savename + "_predictions.txt"
    f = open(result_file, "w+")
    f.close()

    subset_dir = subset if subset == "test" else "train_val"
    for c in cities:
        print(c)
        m_raw_file = root_dir + subset_dir + "/" + c + "/database/raw.csv"
        q_idx_file = root_dir + subset_dir + "/" + c + "/query.json"
        m_idx_file = root_dir + subset_dir + "/" + c + "/database.json"
        q_dl = create_dataloader("test", root_dir, q_idx_file, None, image_t, batch_size)
        q_feats_file = results_dir + "/" + savename + "_" + c + "_queryfeats.npy"
        if cls_token:
            q_feats_cls_file = results_dir + "/" + savename + "_" + c + "_queryfeats_cls.npy"
            extract_features(q_dl, net, f_length, q_feats_file, q_feats_cls_file)
        else:
            extract_features(q_dl, net, f_length, q_feats_file)
        m_dl = create_dataloader("test", root_dir, m_idx_file, None, image_t, batch_size)
        m_feats_file = results_dir + "/" + savename + "_" + c + "_mapfeats.npy"
        if cls_token:
            m_feats_cls_file = results_dir + "/" + savename + "_" + c + "_mapfeats_cls.npy"
            extract_features(m_dl, net, f_length, m_feats_file, m_feats_cls_file)
        else:
            extract_features(m_dl, net, f_length, m_feats_file)
        result_file = extract_msls_top_k(m_feats_file, q_feats_file, m_idx_file, q_idx_file, result_file, k, m_raw_file)
    if subset == "val":
        print(result_file)
        score_file = result_file.replace("_predictions", "_result")
        if not os.path.exists(score_file):
            validate(result_file, root_dir, score_file)


def load_index(index):
    with open(index) as f:
        data = json.load(f)
    im_paths = np.array(data["im_paths"])
    im_prefix = data["im_prefix"]

    if "poses" in data.keys():
        poses = np.array(data["poses"])
        return im_paths, poses, im_prefix
    else:
        return im_paths, im_prefix


def world_to_camera(pose):
    [w_qw, w_qx, w_qy, w_qz, w_tx, w_ty, w_tz] = pose
    r = R.from_quat([w_qx, w_qy, w_qz, w_qw]).as_matrix().T
    tx, ty, tz = np.dot(np.array([w_tx, w_ty, w_tz]), np.linalg.inv(-r))
    qx, qy, qz, qw = R.from_matrix(r).as_quat()
    return qw, qx, qy, qz, tx, ty, tz

def predict_poses_cmu(root_dir, m_feats_file, q_feats_file):
    ref_impaths, ref_poses, ref_impref = load_index(root_dir + "reference.json")
    test_impaths, test_impref = load_index(root_dir + "test.json")
    name = "ExtendedCMU" if "extended" in m_feats_file else "CMU"
    D, I = search(m_feats_file, q_feats_file, 1)
    name = m_feats_file.replace("_mapfeats", "_toeval").replace("/MSLS_", "/" + name + "_eval_MSLS_").replace(".npy",
                                                                                                              ".txt")
    with open(name, "w") as f:
        for q, db_index in tqdm(zip(test_impaths, I), desc="Predicting poses..."):
            cut_place = q.find("/img")
            q_im_tosubmit = q[cut_place + 1:]
            pose = np.array((ref_poses[db_index])).flatten()
            submission = q_im_tosubmit + " " + " ".join(pose.astype(str)) + "\n"
            f.write(submission)


def predict_poses(root_dir, m_feats_file, q_feats_file):
    ref_impaths, ref_poses, ref_impref = load_index(root_dir + "reference.json")
    test_impaths, test_impref = load_index(root_dir + "test.json")

    D, best_score = search(m_feats_file, q_feats_file, 1)

    name = m_feats_file.replace("_mapfeats", "_toeval").replace("/MSLS_", "/RobotCar_eval_MSLS_").replace(".npy",
                                                                                                          ".txt")
    with open(name, "w") as f:
        for q_im, db_index in tqdm(zip(test_impaths, best_score), desc="Predicting poses..."):
            cut_place = q_im.find("/rear")
            q_im_tosubmit = q_im[cut_place + 1:]
            assert q_im_tosubmit.startswith("rear/")
            pose = np.array(world_to_camera(ref_poses[db_index].flatten()))
            submission = q_im_tosubmit + " " + " ".join(pose.astype(str)) + "\n"
            f.write(submission)


def eval_pitts(root_dir, ds, result_file):
    if "pitts" in ds:
        gt_file = root_dir + ds + "_test_gt.h5"
    elif ds.lower() == "tokyotm":
        gt_file = root_dir + "val_gt.h5"
    else:
        gt_file = root_dir + "gt.h5"
    ret_idx = np.load(result_file)
    score_file = result_file.replace("predictions.npy", "scores.txt")
    print(ret_idx.shape)
    ks = [1, 2, 3, 4, 5, 10, 15, 20, 25]
    with open(score_file, "w") as sf:
        with h5py.File(gt_file, "r") as f:
            gt = f["sim"]
            print(gt.shape)
            for k in ks:
                hits = 0
                total = 0
                for q_idx, ret in enumerate(ret_idx):
                    if np.any(gt[q_idx, :]):
                        total += 1
                        db_idx = sorted(ret[:k])
                        hits += np.any(gt[q_idx, db_idx])
                print(k, np.round(hits / total * 100, 2))
                sf.write(str(k) + "," + str(np.round(hits / total * 100, 2)) + "\n")


def extract_features_map_query(root_dir, q_idx_file, m_idx_file, net, f_length, savename, results_dir, batch_size, k,
                               ds):
    q_dl = create_dataloader("test", root_dir, q_idx_file, None, image_t, batch_size)
    q_feats_file = results_dir + "/" + savename + "_queryfeats.npy"
    extract_features(q_dl, net, f_length, q_feats_file)
    m_dl = create_dataloader("test", root_dir, m_idx_file, None, image_t, batch_size)
    m_feats_file = results_dir + "/" + savename + "_mapfeats.npy"
    extract_features(m_dl, net, f_length, m_feats_file)
    result_file = results_dir + "/" + savename + "_predictions.npy"
    if ds.lower() == "tokyotm":
        extract_top_k_tokyotm(m_feats_file, q_feats_file, m_idx_file, q_idx_file, result_file, k)
    else:
        extract_top_k(m_feats_file, q_feats_file, result_file, k)
    if ds == "robotcarseasons":
        predict_poses(root_dir, m_feats_file, q_feats_file)
    elif ds == "extendedcmu" or ds == "cmu":
        predict_poses_cmu(root_dir, m_feats_file, q_feats_file)
    elif "pitts" in ds or "tokyo" in ds:
        eval_pitts(root_dir, ds, result_file)


def extract_top_k_tokyotm(m_feats_file, q_feats_file, db_idx_file, q_idx_file, result_idx_file, k):
    print("TokyoTM")
    D, best_score = search(m_feats_file, q_feats_file)
    with open(db_idx_file, "r") as f:
        db_paths = np.array(json.load(f)["im_paths"])
    with open(q_idx_file, "r") as f:
        q_paths = np.array(json.load(f)["im_paths"])
    result_idx = np.zeros((len(q_paths), k))
    for i, q in enumerate(q_paths):
        q_timestamp = int(q.split("/")[3][1:])
        aux = 0
        for t in range(k):
            idx = best_score[i, aux]
            db = db_paths[idx]
            db_timestamp = int(db.split("/")[3][1:])

            while (np.abs(q_timestamp - db_timestamp) < 1):  # ensure we retrieve something at least a month away
                aux += 1
                idx = best_score[i, aux]
                db = db_paths[idx]
                db_timestamp = int(db.split("/")[3][1:])
            result_idx[i, t] = best_score[i, aux]
            aux += 1

    np.save(result_idx_file, result_idx.astype(int))


def extract_msls_top_k(map_feats_file, query_feats_file, db_idx_file, q_idx_file, result_file, k, m_raw_file=""):
    D, I = search(map_feats_file, query_feats_file, k)

    # load indices
    with open(db_idx_file, "r") as f:
        db_paths = np.array(json.load(f)["im_paths"])
    with open(q_idx_file, "r") as f:
        q_paths = np.array(json.load(f)["im_paths"])
    with open(result_file, "a+") as f:
        for i, q in enumerate(q_paths):
            q_id = q.split("/")[-1].split(".")[0]
            f.write(q_id + " " + " ".join([db_paths[j].split("/")[-1].split(".")[0] for j in I[i, :]]) + "\n")
    return result_file


def search(map_feats_file, query_feats_file, k=25):
    # load features
    query_feats = np.load(query_feats_file).astype('float32')
    map_feats = np.load(map_feats_file).astype('float32')
    if k is None:
        k = map_feats.shape[0]
    # build index and add map features
    index = faiss.IndexFlatL2(map_feats.shape[1])
    index.add(map_feats)
    # search top K
    D, I = index.search(query_feats.astype('float32'), k)
    return D, I


def extract_top_k(map_feats_file, query_feats_file, result_file, k):
    D, I = search(map_feats_file, query_feats_file, k, m)
    np.save(result_file, I)


if __name__ == "__main__":
    p = TestParser()
    p.parse()
    params = p.opt
    print(params)
    # Create model and load weights
    pool = params.pool
    test_net = create_model(params.backbone, pool, norm=params.norm, mode="single")
    try:
        if torch.cuda.is_available():
            test_net.load_state_dict(torch.load(params.model_file)["model_state_dict"])
        else:
            test_net.load_state_dict(
                torch.load(params.model_file, map_location=torch.device('cpu'))["model_state_dict"])
    except:
        if torch.cuda.is_available():
            test_net.load_state_dict(torch.load(params.model_file)["state_dict"])
        else:
            test_net.load_state_dict(torch.load(params.model_file, map_location=torch.device('cpu'))["state_dict"])
    # try:
    #    test_net.load_state_dict(torch.load(params.model_file)["model_state_dict"])
    # except:
    #    test_net.load_state_dict(torch.load(params.model_file)["state_dict"])
    if torch.cuda.is_available():
        test_net.cuda()
    test_net.eval()

    # Create the datasets
    image_size = [int(x) for x in (params.image_size).split(",")]
    if len(image_size) == 2:
        print("testing with images of size", image_size[0], image_size[1])
        image_t = ttf.Compose([ttf.Resize(size=(image_size[0], image_size[1])),
                               ttf.ToTensor(),
                               ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])
    else:
        print("testing with images of size", image_size[0])
        image_t = ttf.Compose([ttf.Resize(size=image_size[0]),
                               ttf.ToTensor(),
                               ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])

    f_length = int(params.f_length)

    results_dir = "results/" + params.dataset + "/" + params.subset + "/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    savename = params.model_file.split("/")[-1].split(".")[0]
    print(savename)
    if params.dataset.lower() == "msls":
        extract_features_msls(params.subset, params.root_dir, test_net, f_length, image_t, savename, results_dir,
                              params.batch_size, 30)
    else:
        extract_features_map_query(params.root_dir, params.query_idx_file, params.map_idx_file, test_net, f_length,
                                   savename, results_dir, params.batch_size, 30, params.dataset.lower())
