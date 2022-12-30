
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

# vit_large_patch14_224_clip_laion2b
# eva_large_patch14_196_in22k_ft_in22k_in1k
# resnet50
model_name = 'vit_large_patch14_224_clip_laion2b'
# import pdb;pdb.set_trace()
model = timm.create_model(model_name, pretrained=True)
model.eval()
model = model.cuda()


# import pdb;pdb.set_trace()

# load the image transformer
t = []
# maintain same ratio w.r.t. 224 images
# follow https://github.com/facebookresearch/mae/blob/main/util/datasets.py
t.append(T.Resize(model.pretrained_cfg['input_size'][1], interpolation=Image.BICUBIC))
t.append(T.CenterCrop(model.pretrained_cfg['input_size'][1]))
t.append(T.ToTensor())
t.append(T.Normalize(model.pretrained_cfg['mean'], model.pretrained_cfg['std']))
center_crop = T.Compose(t)


save_dir = "/mnt/lustre/yhzhang/data/imagenet/features_vit_train-shot16-seed0"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    print(f"Directory exists at {save_dir}")
    sys.exit()


image_root = "/mnt/lustre/share/DSK/datasets/imagenet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train"

meta_path = '/mnt/lustre/yhzhang/data/imagenet/annotations/train_meta.list.num_shot_16.seed_0'

with open(meta_path) as f:
    metas = f.readlines()

examples = [cur_line.strip().split(' ')[0] for cur_line in metas]

imgs = []

global_features = torch.tensor([]).cuda()
for example in tqdm(examples):
    try:
        path = os.path.join(image_root,example)
        img = Image.open(path).convert("RGB")
        img = center_crop(img)
        imgs.append(img)
    except:
        print(f"Disappear {path}")
        sys.stdout.flush()

    if len(imgs) == 128:

        imgs = torch.stack(imgs).cuda()
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            features = model.forward_features(imgs)
            features = model.forward_head(features,pre_logits=True)
            if len(global_features) == 0:
                global_features = features
            else:
                global_features = torch.cat((global_features,features))

        imgs = []

if len(imgs) > 0: 
    imgs = torch.stack(imgs).cuda()
    with torch.no_grad():
        features = model.forward_features(imgs)
        features = model.forward_head(features,pre_logits=True)
        if len(global_features) == 0:
            global_features = features
        else:
            # import pdb;pdb.set_trace()
            global_features = torch.cat((global_features,features))

features = global_features.cpu().numpy().astype(np.float32)

save_file = os.path.join(save_dir, 'features')

np.savez(save_file, examples=examples, features=features)