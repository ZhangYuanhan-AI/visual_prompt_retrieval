import numpy as np
import scipy.spatial.distance as distance
import os
from tqdm import tqdm
import sys
from scipy import linalg, mat, dot
import json



features_dir = "/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/features_rn50_folder_val"

origin_features_files = os.listdir(features_dir)

features_files = []
for i in origin_features_files:
    if not i.endswith('.npz'):
        continue

    features_files.append(i)


for foldid in range(len(features_files)):
    feature_file = 'folder'+str(foldid)+'.npz'
    print(f"Processing {feature_file} ...")
    sys.stdout.flush()
    path = os.path.join(features_dir, feature_file)
    file_npz = np.load(path)
    examples = file_npz["examples"].tolist()
    features = file_npz["features"].astype(np.float32)
    # import pdb;pdb.set_trace() #2007_000648
    similarity = dot(features,features.T)/linalg.norm(features,axis=1)/linalg.norm(features,axis=1)
    for i in range(len(similarity)):
        similarity[i][i] = 0

    similarity_idx = np.argsort(similarity,axis=1)[:,-50:]

    similarity_idx_dict = {}
    for _, (cur_example, cur_similarity) in enumerate(zip(examples,similarity_idx)):
        img_name = cur_example.strip().split('/')[-1][:-4]
        # if img_name == '2008_007883':
        #     import pdb;pdb.set_trace()
        if img_name not in similarity_idx_dict:
            similarity_idx_dict[img_name] = list(examples[idx].strip().split('/')[-1][:-4] for idx in cur_similarity[::-1])
    
    with open(features_dir+'/folder'+str(foldid)+'_top50-similarity'+'.json', "w") as outfile:
        json.dump(similarity_idx_dict, outfile)