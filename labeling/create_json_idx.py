import argparse
import os 
import numpy as np
import json
import transformations as tf

class LabelParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--dataset', required=True, default='MSLS')

        self.parser.add_argument('--root_dir', required=True, help='Root directory of the dataset')
        

    def parse(self):
        self.opt = self.parser.parse_args()

def get_city_idx(root_dir, city, name):

    dire=os.path.join(root_dir,'train_val',city,name)
    db_data_pp=np.genfromtxt(os.path.join(dire,'postprocessed.csv'), dtype="str", delimiter=",",skip_header=1)
    db_data_raw=np.genfromtxt(os.path.join(dire,'raw.csv'), dtype="str", delimiter=",",skip_header=1)
    filenames=[]
    locs=[]
    for i in range(len(db_data_pp)):
        key_pp=db_data_pp[i,1]
        key_raw=db_data_raw[i,1]
        assert key_pp==key_raw
        filename=os.path.join('train_val',city,name,'images/',key_pp+'.jpg')
        assert os.path.exists(os.path.join(root_dir,filename)), filename+" not found"
        east=float(db_data_pp[i,2])
        north=float(db_data_pp[i,3])
        ca=float(db_data_raw[i,4])
        loc=[east,north,ca]
        locs.append(loc)
        filenames.append(filename)
    idx={"im_prefix":root_dir, "im_paths":filenames, "loc":locs}
    idx_file=os.path.join(root_dir,'train_val',city,name+".json")
    with open(idx_file,"w") as f:
        json.dump(idx, f)
    print(idx_file+" created succesfully")


def create_idx_msls(root_dir):
    data_path=os.path.join(root_dir, "train_val")
    for f in sorted(os.listdir(data_path)):
        city_path=os.path.join(data_path,f)
        if os.path.isdir(city_path) and os.path.exists(os.path.join(city_path,'query/postprocessed.csv')):
            print("Creating json index for "+ f)
            get_city_idx(root_dir, f, "query")
            get_city_idx(root_dir, f, "database")
            

def create_idx_scene(root_dir, scene, name):
    scene_path = os.path.join(root_dir, scene)
    split_file=os.path.join(scene_path,name.capitalize()+"Split.txt")
    split_sequences=np.genfromtxt(split_file, dtype="str")
    if (len(split_sequences.shape))==0:
        split_sequences=np.expand_dims(split_sequences, 0)
    print(split_file,split_sequences)
    image_files=[]
    poses=[]
    for s in split_sequences:
        seq_name="seq-"+(s.split("sequence")[-1]).zfill(2)
        seq_dir=os.path.join(scene_path, seq_name)
        frames=np.unique([x.split(".")[0] for x in sorted(os.listdir(seq_dir))])
        for f in frames:
            image_file=f+".color.png"
            image_files.append(os.path.join(scene, seq_name,image_file))
            pose_file=f+".pose.txt"
            pose=np.genfromtxt(os.path.join(seq_dir,pose_file))
            tr=pose[:3,-1]
            rot=tf.quaternion_from_matrix(pose)
            sixdof_pose=list(np.hstack((tr,rot)))
            poses.append(sixdof_pose)
    idx={"im_prefix":root_dir, "im_paths":image_files, "poses":poses}
    idx_file=os.path.join(root_dir,scene,name+".json")
    with open(idx_file,"w") as f:
        json.dump(idx, f)
    print(idx_file+" created succesfully")
    
    

def create_idx_7scenes(root_dir):
    for s in sorted(os.listdir(root_dir)):
        scene_path = os.path.join(root_dir, s)
        if os.path.isdir(scene_path) and os.path.exists(os.path.join(scene_path, "TrainSplit.txt")):
            create_idx_scene(root_dir, s, "train")
            create_idx_scene(root_dir, s, "test")
            
if __name__ == "__main__":
    p = LabelParser()
    p.parse()
    params = p.opt
    if params.dataset.lower() == "msls":
        create_idx_msls(params.root_dir)
    elif params.dataset.lower() == "7scenes":
        create_idx_7scenes(params.root_dir)
  
