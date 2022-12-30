# 图片拼接
from PIL import Image
# pil paste可以进行图片拼接
import cv2
import numpy as np
import os
from tqdm import tqdm
 
for foldid in ["2","3"]:
    root = '/mnt/lustre/yhzhang/visual_prompting/evaluate/output_'+foldid
    if not os.path.exists(root+'_concat'):
        os.makedirs(root+'_concat')

    for i in tqdm(os.listdir(root)):
        same_class = os.path.join(root,i)
        # import pdb;pdb.set_trace()
        random_class = os.path.join(root.replace("output","output_random"),i)
        cluster_class = os.path.join(root.replace("output","output_cluster"),i)
        img_out=cv2.imread(same_class)
        img_tmp=cv2.imread(random_class)
        img_tmp2=cv2.imread(cluster_class)

        #横向
        # import pdb;pdb.set_trace()
        try:
            img_out = np.concatenate((img_out,img_tmp,img_tmp2), axis=1)
            # cv2.imshow("IMG",img_out)
            # import pdb;pdb.set_trace()
            cv2.imwrite(root+'_concat/'+i,img_out)
        # cv2.waitKey(0)
        except:
            print(same_class)