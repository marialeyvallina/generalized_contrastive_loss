import torchvision.transforms as ttf
from factory import *
from scipy.spatial.distance import cdist
from tqdm import tqdm 
import sys
import torch
import os
import argparse

msls_cities = {
        'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
                  "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
                  "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
        'val': ["cph", "sf"],
        'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
    }


class TestParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--dataset', required=True, default='MSLS', help='Name of the dataset [MSLS|7Scenes|TB_Places]')

        self.parser.add_argument('--root_dir', required=True, help='Root directory of the dataset')
        self.parser.add_argument('--subset', required=False, default='val', help='For MSLS. Subset to test')

        self.parser.add_argument('--query_idx_file', type=str, required=False, help='Query idx file, .json')
        self.parser.add_argument('--map_idx_file', type=str, required=False, help='Map idx file, .json')

        self.parser.add_argument('--model_file', type=str, required=True, help='Model file, .pth')
        self.parser.add_argument('--backbone', type=str, default='resnet50', help='which architecture to use. [resnet18, resnet34, resnet50, resnet152, densenet161]')
        self.parser.add_argument('--pool', type=str, required=True, help='pool type', default='avg')
        self.parser.add_argument('--f_length', type=int, default=2048, help='feature length')
        self.parser.add_argument('--image_size', type=str, default="480,640", help='Input size, separated by commas')
        self.parser.add_argument('--norm', type=str, default=None, help='Normalization descriptors')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size')


    def parse(self):
        self.opt = self.parser.parse_args()

def extract_features(dl, net, f_length, feats_file):
    feats = np.zeros((len(dl.dataset), f_length))
    for i, batch in tqdm(enumerate(dl), desc="Extracting features"):
        x = net.forward(batch.cuda())
        feats[i * dl.batch_size:i * dl.batch_size + dl.batch_size] = x.cpu().detach().squeeze(0)
    
    np.save(feats_file, feats)


def distances(query_feats_file, map_feats_file):
    query_feats=np.load(query_feats_file)
    map_feats=np.load(map_feats_file)
    n = len(query_feats)
    m = len(map_feats)
    dists = np.zeros(( n,m), dtype="float16")
    aux = 0
    for i in tqdm(range(m), desc="Calculating distances"):
        dists[:,i] = cdist(map_feats[i:i + 1, :], query_feats).flatten().astype("float16")
        aux += n - 1 - i
    dists= dists.astype("float16")
    dists_file=map_feats_file.replace("_mapfeats.npy", "_distances.npy")
    np.save(dists_file, dists)
    return dists_file
    

def extract_features_msls(subset, root_dir, net, f_length, image_t, savename, results_dir, batch_size, k):
    cities=default_cities[subset]

    result_file=results_dir+"/"+savename+"_predictions.txt"
    f=open(result_file, "w+")
    f.close()
    for c in cities:
        print(c)
        m_raw_file = root_dir+"train_val/"+c+"/database/raw.csv"
        q_idx_file = root_dir+"train_val/"+c+"/query.json"
        m_idx_file = root_dir+"train_val/"+c+"/database.json"
        q_dl = create_dataloader("test", root_dir, q_idx_file, None, image_t, batch_size)
        q_feats_file =results_dir+"/"+savename+"_"+c+"_queryfeats.npy"
        extract_features(q_dl, net, f_length, q_feats_file)
        m_dl = create_dataloader("test", root_dir, m_idx_file, None, image_t, batch_size)
        m_feats_file =results_dir+"/"+savename+"_"+c+"_mapfeats.npy"
        extract_features(m_dl, net, f_length, m_feats_file)
        dists_file=distances(q_feats_file,m_feats_file)
        extract_msls_top_k(dists_file, m_idx_file, q_idx_file, result_file, k, m_raw_file)


def extract_features_map_query(root_dir, q_idx_file, m_idx_file, net, f_length,savename, results_dir,batch_size, k):
    q_dl = create_dataloader("test", root_dir, q_idx_file, None, image_t, batch_size)
    q_feats_file =results_dir+"/"+savename+"_queryfeats.npy"
    extract_features(q_dl, net, f_length, q_feats_file)
    m_dl = create_dataloader("test", root_dir, m_idx_file, None, image_t, batch_size)
    m_feats_file =results_dir+"/"+savename+"_mapfeats.npy"
    extract_features(m_dl, net, f_length, m_feats_file)
    dists_file=distances(q_feats_file,m_feats_file)
    result_file=results_dir+"/"+savename+"_predictions.npy"
    extract_top_k(dists_file, result_file, k)


def extract_msls_top_k(dists_file, db_idx_file, q_idx_file, result_file, k,m_raw_file=""):
    dists=np.load(dists_file)
    if os.path.exists(m_raw_file):
        m_pano=np.genfromtxt(m_raw_file, dtype=bool, skip_header=1, delimiter=",")[:,-1]

        with open(db_idx_file, "r") as f:
            db_paths=np.array(json.load(f)["im_paths"])[np.logical_not(m_pano)]
            db_keys=[x.split("/")[-1].split(".")[0] for x in db_paths]
    else:
        with open(db_idx_file, "r") as f:
            db_paths=np.array(json.load(f)["im_paths"])
            db_keys=[x.split("/")[-1].split(".")[0] for x in db_paths]
    with open(q_idx_file, "r") as f:
        q_keys=[x.split("/")[-1].split(".")[0] for x in json.load(f)["im_paths"]]
    if os.path.exists(m_raw_file):
        dists = dists[:,np.logical_not(m_pano)]
    best_score = np.argsort(dists, axis=1)
    with open(result_file, "a+") as f:
        for i,q in enumerate(q_keys):
            f.write(q+" "+" ".join([db_keys[j] for j in best_score[i,:k]])+"\n")
            
def extract_top_k(dists_file, result_file, k):
    dists=np.load(dists_file)
    
    best_score = np.argsort(dists, axis=1)[:,:k]
    np.save(result_file,best_score)


if __name__ == "__main__":
    p = TestParser()
    p.parse()
    params = p.opt

    #Create model and load weights
    pool=params.pool
    test_net = create_model(params.backbone, pool, norm=params.norm, mode="single")
    try:
        test_net.load_state_dict(torch.load(params.model_file)["model_state_dict"])
    except:
        test_net.load_state_dict(torch.load(params.model_file)["state_dict"])
    test_net.eval()
    test_net.cuda()

    #Create the datasets
    image_size=[int(x) for x in (params.image_size).split(",")]
    if len(image_size)==2:
        print("testing with images of size",image_size[0],image_size[1])
        image_t = ttf.Compose([ttf.Resize(size=(image_size[0],image_size[1])),
                           ttf.ToTensor(),
                           ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           ]) 
    else:
        print("testing with images of size",image_size[0])
        image_t = ttf.Compose([ttf.Resize(size=image_size[0]),
                           ttf.ToTensor(),
                           ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                           ]) 

    f_length = int(params.f_length)

    results_dir = "results/"+params.dataset+"/"+params.subset+"/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    savename=params.model_file.split("/")[-1].split(".")[0]
    if params.dataset.lower() == "msls":
        extract_features_msls(params.subset, params.root_dir, test_net, f_length, image_t, savename, results_dir, params.batch_size, 30)
    else:
        extract_features_map_query(params.root_dir, params.query_idx_file, params.map_idx_file, test_net, f_length, savename, results_dir, params.batch_size, 30)




    