import numpy as np
import h5py
from sklearn.metrics import average_precision_score
import argparse

class EvalParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--prediction_distance_file', required=True, help='File with the query vs map distance matrix. .npy')

        self.parser.add_argument('--gt_file', required=True, help='Ground truth file. .h5')
    def parse(self):
        self.opt = self.parser.parse_args()


def eval_ap(d_file, gt_file):
    distances=np.load(d_file)
    with h5py.File(gt_file, "r") as f:
        gt=f["fov"][:]>0.5
    return average_precision_score(gt.flatten(), distances.flatten()*(-1))


if __name__ == "__main__":
    p = EvalParser()
    p.parse()
    params = p.opt
    print("AP:",eval_ap(params.prediction_distance_file,params.gt_file))
