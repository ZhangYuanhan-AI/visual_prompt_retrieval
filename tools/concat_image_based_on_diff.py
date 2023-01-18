# 图片拼接
from PIL import Image
# pil paste可以进行图片拼接
import cv2
import numpy as np
import os
from tqdm import tqdm

output_list = [
    # 'output'
    # 'output_rn18',
    # 'output_rn50',
    # "output"
    'output',
    'output_vit-in21k-ft-in1k_val',
    'output_vit-eva_val',
    'output_vit_val',
    'output_supcon-vit-laion2b-freeze-encoder_val',
    # 'output_contrastive-in1k-csz224-bsz64-lr0005-ft',
    # 'output_contrastive-in1k-pretrain'
]

root = '/mnt/lustre/yhzhang/visual_prompting/evaluate/output_color_images'
 
for foldid in ["0"]:
    save_dir = f"{root}/output_{foldid}_concat_sup_unsup"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    IoU_dict = {}

    for cur_output in output_list:
        cur_log = f"{root}/{cur_output}_{foldid}/log.txt"

        with open(cur_log) as f:
            metas = f.readlines()

        for cur_line in metas[1:-1]:
            try:
                img_id, sim_id, result = cur_line.split('\t')
            except:
                img_id, result = cur_line.split('\t')
            if img_id not in IoU_dict:
                IoU_dict[img_id] = []
            IoU = round(eval(result)['mse'],4)
            IoU_dict[img_id].append(IoU)

    random_dir = f"{root}/{output_list[0]}_{foldid}"
    vit_dir = f"{root}/{output_list[1]}_{foldid}"
    mim_dir = f"{root}/{output_list[2]}_{foldid}"
    clip_dir = f"{root}/{output_list[3]}_{foldid}"
    retriver_dir = f"{root}/{output_list[4]}_{foldid}"

    for i in tqdm(IoU_dict):  
        # 
        if  min(IoU_dict[i][:-1])-IoU_dict[i][-1] >= 0.05:
            # import pdb;pdb.set_trace()
            random_img = os.path.join(random_dir, 'generated_{}.png'.format(i))
            retriver_img = os.path.join(retriver_dir, 'generated_{}_0.png'.format(i))
            vit_img = os.path.join(vit_dir, 'generated_{}_0.png'.format(i))
            mim_img = os.path.join(mim_dir, 'generated_{}_0.png'.format(i))
            clip_img = os.path.join(clip_dir, 'generated_{}_0.png'.format(i))

            img1=cv2.imread(random_img)
            img2=cv2.imread(vit_img)
            img3=cv2.imread(mim_img)
            img4=cv2.imread(clip_img)
            img5=cv2.imread(retriver_img)

            #横向
            # import pdb;pdb.set_trace()
            # try:
            # img_out = np.concatenate((img_out,img_tmp), axis=1)
            img_out = np.concatenate((img1,img2,img3,img4, img5), axis=1)
            # cv2.imshow("IMG",img_out)
            # import pdb;pdb.set_trace()
            cv2.imwrite(f"{save_dir}/generated_{i}_{IoU_dict[i][0]}_{IoU_dict[i][1]}_{IoU_dict[i][2]}_{IoU_dict[i][3]}_{IoU_dict[i][4]}.png",img_out)
