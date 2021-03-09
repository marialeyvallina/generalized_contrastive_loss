import numpy as np
import h5py
import argparse

class EvalParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--prediction_distance_file', required=True, help='File with the query vs map distance matrix. .npy')
        self.parser.add_argument('--gt_file', required=True, help='Ground truth file. .h5')
        self.parser.add_argument('--ks', required=False, default="1,2,3,4,5,10,15,20,25", help='Ks you want to evaluate, ordered and separated by commas')
        self.parser.add_argument('--savefile', required=False, help='file to save your results')

    def parse(self):
        self.opt = self.parser.parse_args()


def eval_recall(d_file, gt_file, ks):
    best_score=np.argsort(np.load(d_file))
    with h5py.File(gt_file, "r") as f:
    	if "sim" in f.keys():
    		gt=f["sim"][:]>0.5
    	else:
        	gt=f["fov"][:]>0.5
    recalls=[]
    for k in ks:
        retrieved = best_score[:,:k]
        gt_retrieved = np.zeros((len(best_score), k))
        for m in range(len(best_score)):
            gt_retrieved[m,:]=gt[m,retrieved[m,:]]
        hits = np.any(gt_retrieved, axis=1)
        recalls.append( sum(hits)/len(hits))
    return recalls


if __name__ == "__main__":
    p = EvalParser()
    p.parse()
    params = p.opt
    if params.savefile is None:
    	savefile = params.prediction_distance_file.replace("_distances.npy", "_recall.txt")
    ks=[int(k) for k in params.ks.split(",")]
    recall=eval_recall(params.prediction_distance_file,params.gt_file, ks)
    tosave=np.vstack((ks,recall)).T
    np.savetxt(savefile, tosave,delimiter=",", header='k,r')
    print(["K", "Recall"])
    print(tosave)
    print("Results saved to "+savefile)

