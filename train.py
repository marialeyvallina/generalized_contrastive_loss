import torchvision.transforms as ttf
from src.factory import *
import torch
from torch.optim.lr_scheduler import StepLR
from torch import optim
import shutil
from extract_predictions import extract_msls_top_k
from src.validate import validate
from src.criteria import *
import argparse


def extract_features(dl, net, f_length, feats_file):
    feats = np.zeros((len(dl.dataset), f_length))
    for i, batch in tqdm(enumerate(dl), desc="Extracting features"):
        if torch.cuda.is_available():
            x = net.forward_single(batch.cuda())
        else:
            x = net.forward_single(batch)
        feats[i * dl.batch_size:i * dl.batch_size + dl.batch_size] = x.cpu().detach().squeeze(0)
    
    np.save(feats_file, feats)


class TrainParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--root_dir', required=True, help='Root directory of the dataset')
        self.parser.add_argument('--cities', required=False, default='train', help='Subset of MSLS')
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--name', type=str, required=False, help='name of the experiment', default='testexp')
        self.parser.add_argument('--backbone', type=str, default='vgg16', help='which architecture to use. [resnet50, resnet152, resnext, vgg16]')
        self.parser.add_argument('--snapshot_dir', type=str, default='./snapshots', help='models are saved here')
        self.parser.add_argument('--result_dir', type=str, default='./results', help='predictions and results are saved here')
        self.parser.add_argument('--use_gpu', help='Use GPU mode',action='store_true')
        self.parser.add_argument('--save_freq', type=int, default=1, help='save frequency in steps')
        self.parser.add_argument('--dataset', type=str, default='soft_MSLS', help='[binary_MSLS|soft_MSLS]')
        self.parser.add_argument('--pool', type=str, default='GeM', help='Global pool layer  max|avg|GeM')
        self.parser.add_argument('--p', required=False, type=int, default=3, help='P parameter for GeM pool')
        self.parser.add_argument('--norm', type=str, default="L2", help='Norm layer')
        self.parser.add_argument('--image_size', type=str, default="480,640", help='Input size, separated by commas')
        self.parser.add_argument('--last_layer', type=int, default=None, help='Last layer to keep')
        self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results on screen')
        self.parser.add_argument('--steps', type=int, default=52, help='Number of training steps. 52= 1epoch')
        self.parser.add_argument('--margin', type=float, default='.5', help='margin parameter for the contrastive loss')
        self.parser.add_argument('--learning_rate', type=float, default='.1', help='learning rate')
        self.parser.add_argument('--lr_gamma', type=float, default='.1', help='learning rate decay')
        self.parser.add_argument('--step_size', type=float, default='25', help='Learning rate update frequency (in steps)')

    def parse(self):
        self.opt = self.parser.parse_args()


def val(params, model, image_t, best_metric, reference_metric="recall@5", metric="EuclideanDistance"):
    print("Validating...")
    val_cities = ["cph", "sf"]
    save_dir = params.result_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    is_best = False
    model.eval()
    ret_file = save_dir+ "/" + params.name + "_retrieved.csv"
    with open(ret_file, "w"):
        pass
    for c in val_cities:
        print(c)
        m_raw_file = params.root_dir+"train_val/"+c+"/database/raw.csv"
        q_idx_file = params.root_dir+"train_val/"+c+"/query.json"
        m_idx_file = params.root_dir+"train_val/"+c+"/database.json"
        gt_file = params.root_dir+"train_val/"+c+"_gt.h5"
        q_dl = create_dataloader("test", params.root_dir, q_idx_file, None, image_t, 2)
        m_dl = create_dataloader("test", params.root_dir, m_idx_file, None, image_t, 2)

        q_feats_file = save_dir + "/" + params.name + "_"+c+ "_query_features.npy"
        q_feats = extract_features(q_dl, model, model.feature_length,q_feats_file)
        m_feats_file = save_dir + "/" + params.name + "_"+c+ "_database_features.npy"
        m_feats = extract_features(m_dl, model, model.feature_length,m_feats_file)
        extract_msls_top_k(m_feats_file,q_feats_file, m_idx_file, q_idx_file, ret_file, 25, m_raw_file)

    res_file = save_dir + "/" + params.name + "_val_results.txt"
    metrics = validate(ret_file, params.root_dir, res_file)

    if metrics[reference_metric] > best_metric:
        shutil.copy(ret_file, ret_file.replace(".csv", "_best.csv"))
        shutil.copy(res_file, res_file.replace(".txt", "_best.txt"))
        for c in val_cities:
            q_feats_file = save_dir + "/" + params.name + "_"+c+ "_query_features.npy"
            shutil.copy(q_feats_file, q_feats_file.replace(".npy", "_best.npy"))
            m_feats_file = save_dir + "/" + params.name + "_"+c+ "_database_features.npy"
            shutil.copy(m_feats_file, m_feats_file.replace(".npy", "_best.npy"))
        is_best = True
    model.train()
    return metrics, is_best


def train(params):
    image_size = [int(x) for x in (params.image_size).split(",")]
    best_metric = 0
    ref_metric = "recall@5"
    print("training with images of size",image_size[0],image_size[1])
    if image_size[0] == image_size[1]: #If we want to resize to square, we do resize+crop
        image_t = ttf.Compose([ttf.Resize(size=(image_size[0])),
                               ttf.CenterCrop(size=(image_size[0])),
                               ttf.ToTensor(),
                               ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])
    else:        
        image_t = ttf.Compose([ttf.Resize(size=(image_size[0], image_size[1])),
                               ttf.ToTensor(),
                               ttf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                               ])
    # writer = SummaryWriter('runs/'+params.name+"_"+datetime.now().isoformat("-").split(".")[0].replace(":","-"))
    mode = "siamese"
    model = create_model(params.backbone, params.pool, last_layer=params.last_layer, norm=params.norm, p_gem=params.p)
    if torch.cuda.is_available():
        model = model.cuda()
    loss = ContrastiveLoss(params.margin)
    print(params.dataset)
    dataloader = create_msls_dataloader(params.dataset, params.root_dir, params.cities, transform=image_t,
                                        batch_size=params.batch_size, model=model)
    
    # if params.use_gpu:
    if torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()
    total_iterations = 0

    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=params.step_size, gamma=params.lr_gamma)
    init_step = 0
    optimizer.zero_grad()
    # metrics, is_best = val(params, model, image_t, best_metric, reference_metric=ref_metric)
    # best_metric = metrics[ref_metric]
    best_metric = 0
    # for step in tqdm(range(init_step, params.steps), desc="Steps"):
    for step in range(init_step, params.steps):
        # step
        e_iteration = 0
        for i, data in enumerate(dataloader):
            e_iteration += params.batch_size

            mini_batch_size = 2
            accum_iterations = int(data["im0"].shape[0]/mini_batch_size)

            for j in range(accum_iterations):
                a = j * mini_batch_size
                b = a + mini_batch_size

                if torch.cuda.is_available():  # params.use_gpu:
                    x0, x1 = model(data["im0"][a:b,:].cuda(), data["im1"][a:b,:].cuda())
                    error = loss(x0, x1, (data["label"][a:b]).cuda())
                else:
                    x0, x1 = model(data["im0"][a:b,:], data["im1"][a:b,:])
                    error = loss(x0, x1, data["label"][a:b])
                null_losses = torch.sum(error==0).item()/len(error)
                        
                error = torch.mean(error) / accum_iterations
                # writer.add_scalar('Debug/null_losses', null_losses, total_iterations)
                error.backward()
                # writer.add_scalar('Loss/train', error.cpu(), total_iterations)
                total_iterations += mini_batch_size

            # Visualize
            if i % params.display_freq == 0:
                print("Step %d, Iteration %d, Loss %.4f, Null loss %.4f" % (step, e_iteration, error, null_losses))
            optimizer.step()
            optimizer.zero_grad()
            
        metrics, is_best = val(params, model, image_t, best_metric, reference_metric=ref_metric)
        # Save
        if step % params.save_freq == 0:
            save_path = params.snapshot_dir + "/" + params.name + ".pth"
        
            torch.save({'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                        }, save_path)
                
        if is_best:
            best_metric = metrics[ref_metric]
            best_metrics = metrics
            save_path = params.snapshot_dir + "/" + params.name +"_best.pth"
        
            torch.save({'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                        }, save_path)

        scheduler.step()
        dataloader.dataset.load_pairs()
    # writer.close()
    print("Done. Best results on val:")
    print(best_metrics)


if __name__ == "__main__":
    p = TrainParser()
    p.parse()
    params = p.opt
    train(params)
