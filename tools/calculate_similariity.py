import numpy as np
import scipy.spatial.distance as distance
import os
from tqdm import tqdm
import sys
from scipy import linalg, mat, dot
import json



features_dir = "/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/features_vit_folder"

features_files = os.listdir(features_dir)

for foldid, feature_file in enumerate(tqdm(features_files)):
    if not feature_file.endswith('.npz'):
        continue
    print(f"Processing {feature_file} ...")
    sys.stdout.flush()
    path = os.path.join(features_dir, feature_file)
    file_npz = np.load(path)
    examples = file_npz["examples"].tolist()
    features = file_npz["features"].astype(np.float32)
    similarity = dot(features,features.T)/linalg.norm(features,axis=1)/linalg.norm(features,axis=1)
    similarity_idx = np.argsort(similarity,axis=1)[:,-51:-1]

    similarity_idx_dict = {}
    for _ in examples:
        img_name = _.strip().split('/')[-1][:-4]
        if img_name not in similarity_idx_dict:
            similarity_idx_dict[img_name] = list(examples[idx].strip().split('/')[-1][:-4] for idx in similarity_idx[0][::-1])
    
    with open(features_dir+'/folder'+str(foldid)+'_top50-similarity'+'.json', "w") as outfile:
        json.dump(similarity_idx_dict, outfile)
    
