"""
Extract features for SupPR.
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

import timm
from timm.models import load_checkpoint
from collections import OrderedDict

from vit import SupVit



pretrain_name = sys.argv[1]
feature_name = sys.argv[2]
split = sys.argv[3]

def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.','') if 'module.' in k else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict
     
# load the image transformer
t = []
size = 224
t.append(T.Resize((size,size), interpolation=Image.BICUBIC))
t.append(T.CenterCrop(size))
t.append(T.ToTensor())
t.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
center_crop = T.Compose(t)

save_dir = f"/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/{feature_name}_{split}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory exists at {save_dir}")
    sys.exit()


meta_root =  f"/mnt/lustre/yhzhang/visual_prompting/evaluate/splits/pascal/{split}"
image_root = "/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/JPEGImages"


for folder_id in tqdm([0, 1, 2, 3]):

    model = SupVit(pretrain_name)
    ckpt_path = f"/mnt/lustre/yhzhang/SupContrast/save/SupCon/path_models/SupCon_path_{pretrain_name}_folder_{folder_id}_seed_0_lr_0.005_decay_0.0001_cropsz_{size}_bsz_64_temp_0.1_trial_0_cosine_pretrain-vit-freeze-encoder/last.pth"
    try:
        model.load_state_dict(clean_state_dict(torch.load(ckpt_path)['model']))
    except:
        print('{} is wrong'.format(ckpt_path))
        sys.stdout.flush()
        continue
    model.eval()
    model = model.cuda()

    print(f"Processing folder {folder_id}")
    sys.stdout.flush()
    with open(os.path.join(meta_root, 'fold'+str(folder_id)+'.txt')) as f:
        examples = f.readlines()
    if len(examples) == 0:
        print(f"zeros folder{folder_id}")
        sys.stdout.flush()
        continue

    examples = [os.path.join(image_root, example.strip()[:-4]+'.jpg') for example in examples]
       
    imgs = []

    global_features = torch.tensor([]).cuda()
    for example in examples:
        try:
            path = os.path.join(example)
            img = Image.open(path).convert("RGB")
            img = center_crop(img)
            imgs.append(img)
        except:
            print(f"Disappear {path}")
            sys.stdout.flush()

        if len(imgs) == 128:

            imgs = torch.stack(imgs).cuda()
            with torch.no_grad():
                features = model(imgs)
                if len(global_features) == 0:
                    global_features = features
                else:
                    global_features = torch.cat((global_features,features))

            imgs = []

    imgs = torch.stack(imgs).cuda()
    with torch.no_grad():
        features = model(imgs)
        if len(global_features) == 0:
            global_features = features
        else:
            global_features = torch.cat((global_features,features))

    features = global_features.cpu().numpy().astype(np.float32)
    
    save_file = os.path.join(save_dir, 'folder'+str(folder_id))
    np.savez(save_file, examples=examples, features=features)
