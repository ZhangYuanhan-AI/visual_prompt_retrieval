
"""
Extract features using PlacesCNN.
"""
import os
import sys
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

import torch
import torchvision.models as models
from torchvision import transforms as T
from torch.nn import functional as F

from dassl.config import get_cfg_default
from dassl.data.datasets import build_dataset

import timm

sys.path.append('/mnt/lustre/yhzhang/visual_prompting')
from evaluate_detection.voc_orig import VOCDetection 

from resnet import SupConResNet
from vit import SupVit
from collections import OrderedDict


def clean_state_dict(state_dict):
   # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
   cleaned_state_dict = OrderedDict()
   for k, v in state_dict.items():
       name = k.replace('module.','') if 'module.' in k else k
       cleaned_state_dict[name] = v
   return cleaned_state_dict





# load the image transformer
t = []
# maintain same ratio w.r.t. 224 images
# follow https://github.com/facebookresearch/mae/blob/main/util/datasets.py
size = 196
t.append(T.Resize((size,size), interpolation=Image.BICUBIC))
t.append(T.CenterCrop(size))
t.append(T.ToTensor())
t.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
center_crop = T.Compose(t)

model_name = 'eva_large_patch14_196.in22k_ft_in22k_in1k'
model = SupVit(model_name)
# model.load_state_dict(clean_state_dict(torch.load('/mnt/lustre/yhzhang/SupContrast/weights/supcon.pth')['model_ema']))
model.load_state_dict(clean_state_dict(torch.load(f'/mnt/lustre/yhzhang/SupContrast/save/SupCon/path_models/det_SupCon_path_{model_name}_seed_0_lr_0.005_decay_0.0001_cropsz_{size}_bsz_64_temp_0.1_trial_0_cosine_pretrain/last.pth')['model']))
model.eval()
model = model.cuda()


save_dir = f"/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/features_supcon-vit-freeze-encoder-eva-csz{size}-bsz64-lr0005-ft_val_det"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory exists at {save_dir}")
    # sys.exit()

for cur_split in ['val', 'train']:
# ds = VOCDetection('/mnt/lustre/yhzhang/data/pascal-5i/', ['2012'], image_sets=['val'], transforms=None)
    ds = VOCDetection('/mnt/lustre/yhzhang/data/pascal-5i/', ['2012'], image_sets=[cur_split], transforms=None)


    global_features = torch.tensor([]).cuda()
    imgs = []
    examples = []
    for index in tqdm(range(len(ds))):
        try:
            example = ds.images[index]
            # import pdb;pdb.set_trace()
            img = Image.open(example).convert('RGB')
            examples.append(example)
            img = center_crop(img)
            imgs.append(img)
        except:
            print(f"Disappear {ds.images[index]}")
            sys.stdout.flush()

        if len(imgs) == 128:

            imgs = torch.stack(imgs).cuda()
            # import pdb;pdb.set_trace()
            with torch.no_grad():
                features = model(imgs)
                if len(global_features) == 0:
                    global_features = features
                else:
                    global_features = torch.cat((global_features,features))

            imgs = []

    if len(imgs) > 0:
        imgs = torch.stack(imgs).cuda()
        with torch.no_grad():
            features = model(imgs)
            if len(global_features) == 0:
                global_features = features
            else:
                # import pdb;pdb.set_trace()
                global_features = torch.cat((global_features,features))

    features = global_features.cpu().numpy().astype(np.float32)
    save_file = os.path.join(save_dir, 'features_{}'.format(cur_split))
    np.savez(save_file, examples=examples, features=features)