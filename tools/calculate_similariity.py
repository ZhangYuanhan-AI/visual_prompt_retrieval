import numpy as np
import scipy.spatial.distance as distance
import os
from tqdm import tqdm
import sys
from scipy import linalg, mat, dot
import json



features_dir = "/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/features_vit_det"

origin_features_files = os.listdir(features_dir)

features_files = []
for i in origin_features_files:
    if not i.endswith('.npz'):
        continue

    features_files.append(i)


feature_val_file = 'features_train.npz'
feature_train_file = 'features_train.npz'

print(f"Processing {feature_val_file} ...")
sys.stdout.flush()
val_path = os.path.join(features_dir, feature_val_file)
file_val_npz = np.load(val_path)
val_examples = file_val_npz["examples"].tolist()
val_features = file_val_npz["features"].astype(np.float32)

print(f"Processing {feature_train_file} ...")
sys.stdout.flush()
train_path = os.path.join(features_dir, feature_train_file)
file_train_npz = np.load(train_path)
train_examples = file_train_npz["examples"].tolist()
train_features = file_train_npz["features"].astype(np.float32)
# import pdb;pdb.set_trace() #2007_000648
similarity = dot(val_features,train_features.T)/(linalg.norm(val_features,axis=1, keepdims=True) * linalg.norm(train_features,axis=1, keepdims=True).T)

similarity_idx = np.argsort(similarity,axis=1)[:,-50:]

similarity_idx_dict = {}
for _, (cur_example, cur_similarity) in enumerate(zip(val_examples,similarity_idx)):
    img_name = cur_example.strip().split('/')[-1][:-4]
    # if img_name == '2008_007883':
    # import pdb;pdb.set_trace()
    if img_name not in similarity_idx_dict:
        similarity_idx_dict[img_name] = list(train_examples[idx].strip().split('/')[-1][:-4]+' '+str(idx) for idx in cur_similarity[::-1])

print(f"len of similarity is {len(similarity_idx_dict)} ...")
sys.stdout.flush()

with open(features_dir+'/train-top50-similarity'+'.json', "w") as outfile:
    json.dump(similarity_idx_dict, outfile)