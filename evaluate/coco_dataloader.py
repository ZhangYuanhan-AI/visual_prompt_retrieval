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
 
# from https://github.com/matlab-deep-learning/Object-Detection-Using-Pretrained-YOLO-v2/blob/main/%2Bhelper/coco-classes.txt and double check by mmdetection
pascal_5i_coco_train_id = {
    '0': ['4', '1', '14', '8', '39'],
    '1': ['5', '2', '15', '56', '19'],
    '2': ['60', '16', '17', '3', '0'],
    '3': ['58', '18', '57', '6', '62']
}

class DatasetCOCO(Dataset):
    def __init__(self, datapath, fold, transform, split, shot: int = 1, use_original_imgsize: bool = False):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.benchmark = 'coco'
        self.shot = shot
        self.base_path = os.path.join(datapath, 'COCO2014')
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000


    def get_top50_images(self):
        with open('/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/{}/folder{}_top50-similarity_{}_{}.json'.format(self.feature_name, self.fold, self.percentage, self.seed)) as f:
            images_top50 = json.load(f)

        images_top50_new = {}
        for img_name, img_class in self.img_metadata:
            if img_name not in images_top50_new:
                images_top50_new[img_name] = {}

            images_top50_new[img_name]['top50'] = images_top50[img_name]
            images_top50_new[img_name]['class'] = img_class

        return images_top50_new

 
    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask, flip: bool = False):
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding, 2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
        canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
        canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas


    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        grid_stack = torch.tensor([]).cuda()
        # for sim_idx in self.iou_dict[str(idx)]:
            # sim_idx = int(sim_idx)
        for sim_idx in range(1):
            query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

            query_img, query_mask = self.transform(query_img, query_mask)
            query_mask = query_mask.float()
         
            support_transformed = [self.transform(support_img, support_cmask) for support_img, support_cmask in zip(support_imgs, support_masks)]
            support_masks = [x[1] for x in support_transformed]
            support_imgs = torch.stack([x[0] for x in support_transformed])
            
            for midx, smask in enumerate(support_masks):
                support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks = torch.stack(
                
            )

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def build_class_ids(self):
        ## original ##
        # nclass_trn = self.nclass // self.nfolds
        # class_ids_val = [self.fold + self.nfolds * v for v in range(nclass_trn)]
        # class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        # class_ids = class_ids_trn if self.split == 'trn' else class_ids_val

        ## For coco-5i ##
        class_ids = [ int(i) - 1 for i in pascal_5i_coco_train_id[self.fold] ]

        return class_ids

    def build_img_metadata_classwise(self):
        with open('./data/splits/coco/%s/fold%d.pkl' % (self.split, self.fold), 'rb') as f:
            img_metadata_classwise = pickle.load(f)
        return img_metadata_classwise

    def build_img_metadata(self):
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name):
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        query_mask = self.read_mask(query_name)

        org_qry_imsize = query_img.size

        query_mask[query_mask != class_sample + 1] = 0
        query_mask[query_mask == class_sample + 1] = 1

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        support_imgs = []
        support_masks = []
        for support_name in support_names:
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask = self.read_mask(support_name)
            support_mask[support_mask != class_sample + 1] = 0
            support_mask[support_mask == class_sample + 1] = 1
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize
 
 
