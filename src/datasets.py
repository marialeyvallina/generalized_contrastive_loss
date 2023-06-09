from torch.utils.data import Dataset
import json
import h5py
from scipy.spatial.distance import squareform
import torch
import numpy as np
import os
from PIL import Image
import math
import pandas as pd
from tqdm import tqdm

default_cities = {
    'train': ["trondheim", "london", "boston", "melbourne", "amsterdam", "helsinki",
              "tokyo", "toronto", "saopaulo", "moscow", "zurich", "paris", "bangkok",
              "budapest", "austin", "berlin", "ottawa", "phoenix", "goa", "amman", "nairobi", "manila"],
    'toy': ["amman"],
    'val': ["cph", "sf"],
    'test': ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"]
}


class BaseDataSet(Dataset):
    def __init__(self, root_dir, idx_file, gt_file=None, ds_key="sim", transform=None):
        """
        Args:
            idx_file (string): Path to the idx file (.json)
            gt_file (string): Path to the GT file with pairwise similarity (.h5).
            ds_key (string): dataset name
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.im_paths = self.load_idx(idx_file)
        self.root_dir = root_dir
        self.ds_key=ds_key
        if gt_file is not None:
            with h5py.File(gt_file, "r") as f:
                self.gt_matrix = torch.Tensor((f[ds_key][:].flatten()).astype(float))
        self.transform = transform
        self.n = len(self.im_paths)

    @staticmethod
    def load_idx(idx_file):
        with open(idx_file) as f:
            data = json.load(f)
            root_dir = data["im_prefix"]
            im_paths = data["im_paths"]
            return im_paths

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        pass

    def read_image(self, impath):
        img_name = os.path.join(self.root_dir,
                                impath)
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image


class TestDataSet(BaseDataSet):
    def __init__(self, root_dir, idx_file, transform=None):
        super().__init__(root_dir, idx_file, None, transform=transform)

    def __getitem__(self, idx_im):
        return self.read_image(self.im_paths[idx_im])

class SiameseDataSet(BaseDataSet):
    def __init__(self, root_dir, idx_file, gt_file, ds_key="sim", transform=None):
        super().__init__(root_dir, idx_file, gt_file, ds_key, transform)
        self.gt_matrix=squareform(self.gt_matrix)


    def __getitem__(self, idx_im0):
        if self.ds_key=="sim": #binary
            s=np.random.choice([True, False], p=[0.5,0.5])#half positive, half negative
            idx_im1=np.random.choice(np.where(self.gt_matrix[idx_im0,:]==s)[0])
            
        else:#graded
            s=np.random.choice(np.arange(3), p=[0.5,0.25,0.25])#half positive, quarter soft negative, quarter hard negative
            if s==0:
                idx_im1=np.random.choice(np.where(self.gt_matrix[idx_im0,:]>0.5)[0])
            elif s==1:
                idx_im1=np.random.choice(np.where(np.logical_and(self.gt_matrix[idx_im0,:]<0.5,self.gt_matrix[idx_im0,:]>0))[0])
            else:
                idx_im1=np.random.choice(np.where(self.gt_matrix[idx_im0,:]==0)[0])

        sel_label=self.gt_matrix[idx_im0,idx_im1]
        return {"im0": self.read_image(self.im_paths[idx_im0]),
                "im1": self.read_image(self.im_paths[idx_im1]),
                "label": float(sel_label)}


class SiameseDataSetByPairs(BaseDataSet):
    def __init__(self, root_dir, idx_file, gt_file, ds_key="sim", transform=None):
        super().__init__(root_dir, idx_file, gt_file, ds_key, transform)

    
    def __len__(self):
        return len(self.gt_matrix)

    def __getitem__(self, idx_pair):
        idx_im0, idx_im1 = self.condensed_to_square(idx_pair, self.n)
        sel_label = self.gt_matrix[idx_pair]
        return {"im0": self.read_image(self.im_paths[idx_im0]),
                "im1": self.read_image(self.im_paths[idx_im1]),
                "label": float(sel_label)}


class MSLSDataSet(Dataset):
    def __init__(self, root_dir, cities, ds_key="fov", transform=None, cache_size=10000, mode="train", daynight=False):
        self.cities = default_cities[cities]
        self.root_dir = root_dir
        self.transform=transform
        self.ds_key = ds_key
        self.start=0
        self.cache_size=cache_size
        self.daynight=daynight
        if mode=="train":
            self.load_cities()
            if self.daynight:
                self.load_daynight_idcs()
                self.load_pairs_daynight()
            else:
                self.load_pairs()

    def load_cities(self):
        self.total=0
        self.city_starting_idx={}
        
        for c in self.cities:
            gt_file = self.root_dir +"train_val/"+ c + "_gt.h5"
            with h5py.File(gt_file, "r") as f:
                gt = f["fov"]
                self.city_starting_idx[self.total]=c
                self.total+=gt.shape[0]
        print(self.city_starting_idx)


    def find_city(self, idx, city_starting_idx, total):
        starting_list=list(city_starting_idx.keys())
        #we figure out which one is our starting city
        st=starting_list[0]
        c=city_starting_idx[st]
        next_st=starting_list[0]
        for c_i in range(1,len(city_starting_idx.keys())):

                if idx >= starting_list[c_i-1] and idx < starting_list[c_i]:
                    st=starting_list[c_i-1]
                    c=city_starting_idx[st]
                    next_st=starting_list[c_i]
                    break
                st=starting_list[c_i]
                c=city_starting_idx[st]
                next_st=total
        return st, c, next_st

    def load_database_cache(self, nDescriptors=50000, nPerImage=100):
        db_total=0
        db_city_starting_idx={}
        
        for c in self.cities:
            gt_file = self.root_dir +"train_val/"+ c + "_gt.h5"
            with h5py.File(gt_file, "r") as f:
                gt = f["fov"]
                db_city_starting_idx[db_total]=c
                db_total+=gt.shape[1]
        print(db_city_starting_idx)

        #Determine how many images we're going to need
        nIm=nDescriptors/nPerImage
        db_idcs=sorted(torch.randperm(db_total,dtype=torch.int)[:int(nIm)])
        #Get those images
        st, c,  next_st= self.find_city(db_idcs[0], db_city_starting_idx, db_total)
        mfile = self.root_dir +"train_val/"+ c + "/database.json"
        m_ims = BaseDataSet.load_idx(mfile)
        if "_toy" in self.root_dir:
            m_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/database/raw_train.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
        else:
            m_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/database/raw.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
        cluster_ims=[]
        for idx in tqdm(db_idcs, desc="loading clustering cache"):
            city_qidx=idx-st
            if idx>=next_st:
                st, c,  next_st= self.find_city(idx, db_city_starting_idx,db_total)
                mfile = self.root_dir +"train_val/"+ c + "/database.json"
                m_ims = BaseDataSet.load_idx(mfile)
                if "_toy" in self.root_dir:
                    m_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/database/raw_train.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
                else:
                    m_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/database/raw.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
            city_qidx=idx-st
            while m_pano[city_qidx]:#if we get a panorama we choose another image from the same city
                city_qidx=np.random.choice(range(len(m_ims)))
            cluster_ims.append(m_ims[city_qidx])
        return cluster_ims

    def load_daynight(self, file):
        data=pd.read_csv(file)
        return np.array(data["n2d"]==True),np.array(data["d2n"]==True)
    

    def load_daynight_idcs(self):
        self.d2n_idcs=[]
        self.n2d_idcs=[]

        for c in self.cities:
            n2d,d2n=(self.load_daynight(self.root_dir+"train_val/"+ c + "/query/subtask_index.csv"))
            self.d2n_idcs.extend(d2n)
            self.n2d_idcs.extend(n2d)
        self.d2n_idcs=np.where(self.d2n_idcs)[0]
        self.n2d_idcs=np.where(self.n2d_idcs)[0]

    def load_pairs_daynight(self):
        #if self.start==0:
        all_idcs=torch.randperm(self.total,dtype=torch.int)
        self.idcs=np.hstack((self.n2d_idcs,np.random.choice(self.d2n_idcs,len(self.n2d_idcs)//2), all_idcs[:len(self.n2d_idcs)//2]))
        self.idcs=self.idcs[torch.randperm(len(self.idcs))]
        self.queries = []
        self.matches = []
        self.sims = []
        queries_night=0
        matches_night=0
        #we get the next chunk and sort it (so that we can go by city)
        cached_idcs=sorted(self.idcs)
        samples=np.random.choice(np.arange(3), p=[0.5,0.25,0.25],size=len(cached_idcs))
        st, c,  next_st= self.find_city(cached_idcs[0], self.city_starting_idx, self.total)
        qfile = self.root_dir +"train_val/"+ c + "/query.json"
        mfile = self.root_dir +"train_val/"+ c + "/database.json"
        gt_file = self.root_dir +"train_val/"+ c + "_gt.h5"
        map_daynight,_=self.load_daynight(self.root_dir+"train_val/"+ c + "/database/subtask_index.csv")
        f= h5py.File(gt_file, "r") 
        q_ims = BaseDataSet.load_idx(qfile)
        m_ims = BaseDataSet.load_idx(mfile)
        m_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/database/raw.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
        q_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/query/raw.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
        total_daynights=0
        for idx in tqdm(cached_idcs, desc="loading cache"):
            query_is_night=1 if idx in self.n2d_idcs else 0
            
            city_qidx=idx-st
            if idx>=next_st:
                f.close()
                st, c,  next_st= self.find_city(idx, self.city_starting_idx, self.total)
                gt_file = self.root_dir +"train_val/"+ c + "_gt.h5"
                map_daynight,_=self.load_daynight(self.root_dir+"train_val/"+ c + "/database/subtask_index.csv")

                f= h5py.File(gt_file, "r") 
                qfile = self.root_dir +"train_val/"+ c + "/query.json"
                mfile = self.root_dir +"train_val/"+ c + "/database.json"
                q_ims = BaseDataSet.load_idx(qfile)
                m_ims = BaseDataSet.load_idx(mfile)
                m_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/database/raw.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
                q_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/query/raw.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]


            city_qidx=idx-st
            if not q_pano[city_qidx]: #we skip panoramas
                q_fovs = torch.Tensor(f[self.ds_key][city_qidx,:])
                if self.ds_key=="fov":
                    s=np.random.choice(np.arange(3), p=[0.5,0.25,0.25])
                    if idx in self.d2n_idcs:##We try to select a night match
                        if s==0: #positive
                            idcs=np.where(np.logical_and(map_daynight,np.logical_and(q_fovs >= 0.5, np.logical_not(m_pano))))[0]
                        elif s==1: #soft negative
                            idcs=np.where(np.logical_and(map_daynight,np.logical_and(q_fovs < 0.5, np.logical_and(q_fovs > 0, np.logical_not(m_pano)))))[0]
                        elif s==2: #hard negative
                            idcs=np.where(np.logical_and(map_daynight,np.logical_and(q_fovs == 0, np.logical_not(m_pano))))[0]
                        if len(idcs) > 0:
                            total_daynights+=1
                            matches_night+=1
                            queries_night+=query_is_night
                            match_idx=np.random.choice(idcs)
                            self.queries.append(q_ims[city_qidx])
                            self.matches.append(m_ims[match_idx])
                            self.sims.append(q_fovs[match_idx])
                    else:
                        if s==0: #positive
                            idcs=np.where(np.logical_and(q_fovs >= 0.5, np.logical_not(m_pano)))[0]
                        elif s==1: #soft negative
                            idcs=np.where(np.logical_and(q_fovs < 0.5, np.logical_and(q_fovs > 0, np.logical_not(m_pano))))[0]
                        elif s==2: #hard negative
                            idcs=np.where(np.logical_and(q_fovs == 0, np.logical_not(m_pano)))[0]
                        if len(idcs) > 0:
                            match_idx=np.random.choice(idcs)
                            queries_night+=(query_is_night)

                            self.queries.append(q_ims[city_qidx])
                            self.matches.append(m_ims[match_idx])
                            self.sims.append(q_fovs[match_idx])
                else:
                    s=np.random.choice(np.arange(2))

                    if s==0: #positive
                        idcs=np.where(np.logical_and(q_fovs == 1, np.logical_not(m_pano)))[0]
                    elif s==1: #negative
                        idcs=np.where(np.logical_and(q_fovs == 0, np.logical_not(m_pano)))[0]
                    if len(idcs) > 0:
                        match_idx=np.random.choice(idcs)


                        self.queries.append(q_ims[city_qidx])
                        self.matches.append(m_ims[match_idx])
                        self.sims.append(q_fovs[match_idx])
        f.close()
        #self.start+=self.cache_size
        #if self.start>=len(self.idcs):
            #print(self.start, self.idcs)
        self.start=0
        self.queries = np.asarray(self.queries)
        self.matches = np.asarray(self.matches)
        self.sims = np.asarray(self.sims)
        print(len(self.queries), len(self.matches), len(self.sims))
        print(queries_night, matches_night)
        assert len(self.queries) == len(self.matches) == len(self.sims)

        

    def load_pairs(self):
        if self.start==0:
            self.idcs=torch.randperm(self.total,dtype=torch.int)
        #this every cache_size iters
        self.queries = []
        self.matches = []
        self.sims = []
        #we get the next chunk and sort it (so that we can go by city)
        cached_idcs=sorted(self.idcs[self.start:self.start+self.cache_size])
        samples=np.random.choice(np.arange(3), p=[0.5,0.25,0.25],size=len(cached_idcs))
        st, c,  next_st= self.find_city(cached_idcs[0], self.city_starting_idx, self.total)
        qfile = self.root_dir +"train_val/"+ c + "/query.json"
        mfile = self.root_dir +"train_val/"+ c + "/database.json"
        gt_file = self.root_dir +"train_val/"+ c + "_gt.h5"
        
        f= h5py.File(gt_file, "r") 
        q_ims = BaseDataSet.load_idx(qfile)
        m_ims = BaseDataSet.load_idx(mfile)
        if "_toy" in self.root_dir:
            m_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/database/raw_train.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
            q_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/query/raw_train.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]

        else:
            m_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/database/raw.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
            q_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/query/raw.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]

        for idx in tqdm(cached_idcs, desc="loading cache"):

            
            city_qidx=idx-st
            if idx>=next_st:
                f.close()
                st, c,  next_st= self.find_city(idx, self.city_starting_idx, self.total)
                gt_file = self.root_dir +"train_val/"+ c + "_gt.h5"

                f= h5py.File(gt_file, "r") 
                qfile = self.root_dir +"train_val/"+ c + "/query.json"
                mfile = self.root_dir +"train_val/"+ c + "/database.json"
                q_ims = BaseDataSet.load_idx(qfile)
                m_ims = BaseDataSet.load_idx(mfile)
                if "_toy" in self.root_dir:
                    m_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/database/raw_train.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
                    q_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/query/raw_train.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
                else:
                    m_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/database/raw.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
                    q_pano=np.genfromtxt(self.root_dir +"train_val/"+ c + "/query/raw.csv", dtype=bool, skip_header=1, delimiter=",")[:,-1]
            city_qidx=idx-st
            if not q_pano[city_qidx]: #we skip panoramas
                q_fovs = torch.Tensor(f[self.ds_key][city_qidx,:])
                if self.ds_key=="fov":
                    s=np.random.choice(np.arange(3), p=[0.5,0.25,0.25])
                    if s==0: #positive
                        idcs=np.where(np.logical_and(q_fovs >= 0.5, np.logical_not(m_pano)))[0]
                    elif s==1: #soft negative
                        idcs=np.where(np.logical_and(q_fovs < 0.5, np.logical_and(q_fovs > 0, np.logical_not(m_pano))))[0]
                    elif s==2: #hard negative
                        idcs=np.where(np.logical_and(q_fovs == 0, np.logical_not(m_pano)))[0]
                    if len(idcs) > 0:
                        match_idx=np.random.choice(idcs)
                        self.queries.append(q_ims[city_qidx])
                        self.matches.append(m_ims[match_idx])
                        self.sims.append(q_fovs[match_idx])
                else:
                    s=np.random.choice(np.arange(2))
                    
                    if s==0: #positive
                        idcs=np.where(np.logical_and(q_fovs == 1, np.logical_not(m_pano)))[0]
                    elif s==1: #negative
                        idcs=np.where(np.logical_and(q_fovs == 0, np.logical_not(m_pano)))[0]
                    if len(idcs) > 0:
                        match_idx=np.random.choice(idcs)
                        self.queries.append(q_ims[city_qidx])
                        self.matches.append(m_ims[match_idx])
                        self.sims.append(q_fovs[match_idx])
        f.close()
        self.start+=self.cache_size
        if self.start>=self.total:
            self.start=0
        self.queries = np.asarray(self.queries)
        self.matches = np.asarray(self.matches)
        self.sims = np.asarray(self.sims)
        print(len(self.queries), len(self.matches), len(self.sims))
        assert len(self.queries) == len(self.matches) == len(self.sims)

    def __len__(self):
        return len(self.queries)

    def read_image(self, impath):
        
        img_name = os.path.join(self.root_dir,
                                impath)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image

    def __getitem__(self, idx):

        sample = {"im0": self.read_image(self.queries[idx]), "im1": self.read_image(self.matches[idx]),
                  "label": self.sims[idx]}
        return sample

class ListImageDataSet(BaseDataSet):
    def __init__(self, image_list, transform=None, root_dir=None):
        self.im_paths=image_list
        self.transform=transform
        self.root_dir=root_dir

    def __getitem__(self, idx_im):
        return self.read_image(self.root_dir+self.im_paths[idx_im])



