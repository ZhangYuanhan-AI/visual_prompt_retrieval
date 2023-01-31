import numpy as np
import scipy.spatial.distance as distance
import os
from tqdm import tqdm
import sys
from scipy import linalg, mat, dot
import json


features_name = sys.argv[1]
source_split = sys.argv[1]
target_split = sys.argv[2]

print(f"Processing {features_name} ...")
sys.stdout.flush()

source_features_dir = f"/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/{features_name}_{source_split}"
target_features_dir = f"/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/{features_name]}_{target_split}"

for foldid in [0, 1, 2 ,3]:
    feature_file = 'folder'+str(foldid)+'.npz'
    print(f"Processing {feature_file} ...")
    sys.stdout.flush()
    source_path = os.path.join(source_features_dir, feature_file)
    target_path = os.path.join(target_features_dir, feature_file)
    try:
        source_file_npz = np.load(source_path)
        target_file_npz = np.load(target_path)
    except:
        print(f"no folder {feature_file} ...")
        sys.stdout.flush()
        continue
    source_examples = source_file_npz["examples"].tolist()
    target_examples = target_file_npz["examples"].tolist()
    source_features = source_file_npz["features"].astype(np.float32)
    target_features = target_file_npz["features"].astype(np.float32)

    target_sample_idx = np.random.choice(target_features.shape[0], size=int(target_features.shape[0]), replace=False)
    target_sample_feature = target_features[target_sample_idx,:]
    similarity = dot(source_features,target_sample_feature.T)/(linalg.norm(source_features,axis=1, keepdims=True) * linalg.norm(target_sample_feature,axis=1, keepdims=True).T)

    similarity_idx = np.argsort(similarity,axis=1)[:,-200:]

    similarity_idx_dict = {}
    for _, (cur_example, cur_similarity) in enumerate(zip(source_examples,similarity_idx)):
        img_name = cur_example.strip().split('/')[-1][:-4]

        cur_similar_name = list(target_examples[target_sample_idx[idx]].strip().split('/')[-1][:-4] for idx in cur_similarity[::-1])
        cur_similar_name =  list(dict.fromkeys(cur_similar_name))

        assert len(cur_similar_name) >= 50, "num of cur_similar_name is too small, please enlarge the similarity_idx size"

        if img_name not in similarity_idx_dict:
            similarity_idx_dict[img_name] = cur_similar_name[:50]

    with open(f"{source_features_dir}/folder{foldid}_top50-similarity.json", "w") as outfile:
        json.dump(similarity_idx_dict, outfile)
        
