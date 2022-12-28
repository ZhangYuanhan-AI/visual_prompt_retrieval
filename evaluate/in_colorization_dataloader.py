"""Based on https://github.com/Seokju-Cho/Volumetric-Aggregation-Transformer/blob/main/data/pascal.py
"""
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import json


class DatasetColorization(Dataset):
    def __init__(self, datapath, image_transform, mask_transform, padding: bool = 1,
                 use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, split: str = 'train'):
        self.padding = padding
        self.random = random
        self.use_original_imgsize = use_original_imgsize
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform
        self.datapath = datapath = os.path.join(datapath, split)
        self.ds = self.build_img_metadata(os.path.join(datapath, 'annotations/train_meta.list.num_shot_16.seed_0'))
        # self.ds = self.build_img_metadata(os.path.join(datapath, 'meta/{}.txt'.format(split)))
        self.flipped_order = flipped_order
        np.random.seed(5)
        self.indices = np.random.choice(np.arange(0, len(self.ds)-1), size=1000, replace=False)

        self.image_top50 = self.get_top50_images()


    def __len__(self):
        return 1000

    def get_top50_images(self):
        with open('/mnt/lustre/yhzhang/data/imagenet/features_vit_val/top50-similarity.json') as f:
            images_top50 = json.load(f)

        return images_top50

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask):
        if self.reverse_support_and_query:
            support_img, support_mask, query_img, query_mask = query_img, query_mask, support_img, support_mask
        canvas = torch.ones((support_img.shape[0], 2 * support_img.shape[1] + 2 * self.padding,
                             2 * support_img.shape[2] + 2 * self.padding))
        canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
        if self.flipped_order:
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = query_img
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = support_mask
        else:
            canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
            canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
            canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask

        return canvas

    def build_img_metadata(self, meta_path):

        def read_metadata(meta_path):
            with open(meta_path)) as f:
                metas = f.readlines()
            fold_n_metadata = [cur_line.strip().split(' ')[0] for cur_line in metas]
            return fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata(datapath)

        print('Total %s images are : %d' % (datapath,len(img_metadata)))

        return img_metadata

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.datapath, img_name))

    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        idx = self.indices[idx]
        query = self.ds[idx]

        grid_stack = torch.tensor([]).cuda()
        for sim_idx in range(50):
            support = self.image_top50[query[:-5]][0]+'.JPEG'
            query_img, query_mask = self.mask_transform(self.read_img(query)), self.image_transform(self.read_img(query))
            support_img, support_mask = self.mask_transform(self.read_img(support)), self.image_transform(self.read_img(support))
            grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask)
            if len(grid_stack) == 0:
                grid_stack = grid
            else:
                grid_stack = torch.cat((grid_stack,grid))
            

        batch = {'query_img': query_img, 'query_mask': query_mask, 'support_img': support_img,
                    'support_mask': support_mask, 'grid_stack': grid_stack}

        return batch

if __name__ == "__main__":

    import torchvision

    padding = 1

    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.CenterCrop((224 // 2 - padding, 224 // 2 - padding)),
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.CenterCrop((224 // 2 - padding, 224 // 2 - padding)),
         torchvision.transforms.Grayscale(3),
         torchvision.transforms.ToTensor()])

    canvas_ds = DatasetColorization("/mnt/lustre/yhzhang/data/imagenet", image_transform, mask_transform)


    idx = np.random.choice(np.arange(len(canvas_ds)))

    canvas = canvas_ds[idx]

