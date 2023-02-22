"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset
from mae_utils import PURPLE, YELLOW
import json
 
def create_grid_from_images_old(canvas, support_img, support_mask, query_img, query_mask):
   canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
   canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
   canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
   canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
   return canvas
 
class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, split, image_transform, mask_transform, padding: bool = 1, use_original_imgsize: bool = False, flipped_order: bool = False,
                reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False, purple: bool = False, cluster: bool = False, feature_name: str='features_vit_dino_val', percentage: str='', seed: int=0):
        self.fold = fold
        self.split = split
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20 #20
        self.ncluster = 200
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.cluster = cluster
        self.use_original_imgsize = use_original_imgsize
 
        self.img_path = os.path.join(datapath, 'VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOC2012/SegmentationClassAug/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
 
        self.class_ids = self.build_class_ids()
        self.img_metadata_val = self.build_img_metadata('val') if '_val' in feature_name else self.build_img_metadata('trn') 
        self.img_metadata_trn = self.build_img_metadata('trn')
        self.feature_name = feature_name
        self.seed = seed
        self.percentage = percentage
        self.images_top50_val = self.get_top50_images_val()
        self.images_top50_trn = self.get_top50_images_trn()
         
 
    def __len__(self):
        return 1000 
 
    def get_top50_images_val(self):
        with open(f"/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/{self.feature_name}/folder{self.fold}_top50-similarity.json") as f:
            images_top50 = json.load(f)

        images_top50_new = {}
        for img_name, img_class in self.img_metadata_val:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['top50'] = images_top50[img_name]
            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

    def get_top50_images_trn(self):
        images_top50_new = {}
        for img_name, img_class in self.img_metadata_trn:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['class'] = img_class

        return images_top50_new


    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask, flip: bool = False):
        if self.reverse_support_and_query:
            support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding, 2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if flip:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
 
        return canvas
 
    def __getitem__(self, idx):
        idx %= len(self.img_metadata_val)  # for testing, as n_images < 1000
        grid_stack = torch.tensor([]).cuda()
        # for sim_idx in self.iou_dict[str(idx)]:
            # sim_idx = int(sim_idx)
        for sim_idx in range(1):
            query_name, support_name, class_sample_query, class_sample_support = self.sample_episode(idx,sim_idx)
            query_img, query_cmask, support_img, support_cmask, org_qry_imsize = self.load_frame(query_name, support_name)
            if self.image_transform:
                query_img = self.image_transform(query_img)
                query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask, class_sample_query, purple=self.purple)
            if self.mask_transform:
                query_mask = self.mask_transform(query_mask)
                  
            if self.image_transform:
                support_img = self.image_transform(support_img)
            support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmask, class_sample_support, purple=self.purple)
            if self.mask_transform:
                support_mask = self.mask_transform(support_mask)
            
            
            grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask, flip=self.flipped_order)

            grid = grid.unsqueeze(0)

            if len(grid_stack) == 0:
                grid_stack = grid
            else:
                grid_stack = torch.cat((grid_stack,grid))

        batch = {'query_img': query_img,
                'query_mask': query_mask,
                'query_name': query_name,
                'query_ignore_idx': query_ignore_idx,
                'org_query_imsize': org_qry_imsize,
                'support_img': support_img,
                'support_mask': support_mask,
                'support_name': support_name,
                'support_ignore_idx': support_ignore_idx,
                'class_id': torch.tensor(class_sample_query),
                'grid_stack': grid_stack}
 
        return batch
 
    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x,y] != class_id + 1:
                    color_mask[x, y] = np.array(PURPLE)
                else:
                    color_mask[x, y] = np.array(YELLOW)
        return Image.fromarray(color_mask), boundary
    
    
    def load_frame(self, query_name, support_name):
        # import pdb;pdb.set_trace()
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_img = self.read_img(support_name)
        support_mask = self.read_mask(support_name)
        org_qry_imsize = query_img.size
    
        return query_img, query_mask, support_img, support_mask, org_qry_imsize
    
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask
    
    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')
    
    def sample_episode(self, idx, sim_idx):
        """Returns the index of the query, support and class."""
        if self.cluster:
            query_name, class_sample, cluster_sample = self.img_metadata_val[idx]
        else:
            query_name, class_sample = self.img_metadata_val[idx]
    
        # import pdb;pdb.set_trace()
        if self.random:
            support_class = np.random.choice([k for k in self.img_metadata_classwise.keys() if self.img_metadata_classwise[k]], 1, replace=False)[0]
         
        support_name = self.images_top50_val[query_name]['top50'][sim_idx]
        support_class = self.images_top50_trn[support_name]['class']

        if support_name == query_name:
            print('support_name = query_name ' + support_name)
            return self.sample_episode(idx, sim_idx+1)
        

        return query_name, support_name, class_sample, support_class
    
    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val
    
    def build_img_metadata(self,split):
    
        def read_metadata(split, fold_id):
            cwd = os.path.dirname(os.path.abspath(__file__))
            if self.cluster:
                fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold_cluster%d.txt' % (split, fold_id))
            else:
                fold_n_metadata_path = os.path.join(cwd, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))
    
            with open(fold_n_metadata_path, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            # import pdb;pdb.set_trace()
            if self.cluster:
                fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1, int(data.split('__')[2]) - 1] for data in fold_n_metadata]
            else:
                fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            
            return fold_n_metadata
    
        img_metadata = []
        img_metadata = read_metadata(split, self.fold)
        
        print('Total (%s) images are : %d' % (split,len(img_metadata)))
    
        return img_metadata
    
    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []
    
        if len(self.img_metadata[0]) != 3:
            for img_name, img_class in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]
        else:
            for img_name, img_class, _ in self.img_metadata:
                img_metadata_classwise[img_class] += [img_name]
    
        return img_metadata_classwise
    

 
 
 

