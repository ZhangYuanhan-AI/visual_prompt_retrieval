
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


# vit_large_patch14_224_clip_laion2b
# eva_large_patch14_196.in22k_ft_in22k_in1k
# resnet50
# vit_large_patch16_224.augreg_in21k_ft_in1k
# resnet18
# vit_large_patch14_clip_224.laion2b_ft_in12k_in1k
# vit_base_patch16_224.dino
# import pdb;pdb.set_trace()
model_list = sys.argv[1:]
# ['resnet50','eva_large_patch14_196.in22k_ft_in22k_in1k','vit_large_patch16_224.augreg_in21k_ft_in1k', 'resnet18', 'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k', 'vit_base_patch16_224.dino']:
features_name_dict = {
    'resnet50': 'features_rn50_val_det',
    'eva_large_patch14_196.in22k_ft_in22k_in1k': 'features_vit-eva_val_det',
    'vit_large_patch16_224.augreg_in21k_ft_in1k': 'features_vit-in21k-ft-in1k_val_det',
    'resnet18': 'features_rn18_val_det',
    'vit_large_patch14_clip_224.laion2b_ft_in12k_in1k': 'features_vit-laion2b-ft-in12k-in1k_val_det',
    'vit_base_patch16_224.dino': 'features_vit-dino_val_det'
}

for model_name in model_list:
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


    save_dir = "/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/{}".format(features_name_dict[model_name])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(f"Directory exists at {save_dir}")
        sys.exit()

    for cur_split in ['train','val']:
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
        save_file = os.path.join(save_dir, 'features_{}'.format(cur_split))
        np.savez(save_file, examples=examples, features=features)