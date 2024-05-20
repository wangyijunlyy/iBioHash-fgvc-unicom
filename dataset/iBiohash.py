# https://github.com/elias-ramzi/ROADMAP/blob/main/roadmap/datasets/base_dataset.py
# https://github.com/elias-ramzi/ROADMAP/blob/main/roadmap/datasets/inaturalist.py

import json
from collections import Counter
from os.path import join
import numpy as np, os, sys, pandas as pd, csv, copy
import torch
from PIL import Image
import os
from torch.utils.data import Dataset

class IBioHash(Dataset):

    def __init__(self, root, mode, transform = None):
        self.root = root 
        self.mode = mode
        self.transform = transform

        train_data = pd.read_csv(os.path.join(root, 'train.csv'))     
        gallery_data = pd.read_csv('/media/wsco/linux_gutai2/ibiohash/gallery_images.csv')       
        query_data = pd.read_csv('/media/wsco/linux_gutai2/ibiohash/query_images.csv')
        
        if self.mode == 'train':
            # num_train_samples = len(train_data) // 80
            # selected_indices = np.random.choice(len(train_data), num_train_samples, replace=False)
            self.im_paths = train_data['image_id'].to_numpy()
            self.ys = train_data['label'].to_numpy()
        elif self.mode == 'query':
            self.im_paths = query_data['image_id'].to_numpy()
            self.ys = query_data['label'].to_numpy()
        elif self.mode == 'gallery':
            self.im_paths = gallery_data['image_id'].to_numpy()
            self.ys = gallery_data['label'].to_numpy()

    def nb_classes(self):
        return len(set(self.ys))
            
    def __len__(self):
        print(len(self.ys))
        return len(self.ys)
            
    def __getitem__(self, index):
        
        def img_load(index):
            if self.mode == 'query':
                im = Image.open(os.path.join(self.root, 'Query/Query', self.im_paths[index])).convert('RGB')
            elif self.mode == 'gallery':
                im = Image.open(os.path.join(self.root, 'Gallery/Gallery', self.im_paths[index])).convert('RGB')
            else:
                im = Image.open(os.path.join(self.root, 'iBioHash_Train', self.im_paths[index])).convert('RGB') 
            if self.transform is not None:
                im = self.transform(im)
            return im
        
        im = img_load(index)
        target = self.ys[index]
        image_id = self.im_paths[index]
        
        return im, target , image_id
