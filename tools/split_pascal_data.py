from tqdm import tqdm
import os
import shutil

root = '/mnt/lustre/yhzhang/data/pascal-5i/VOC2012/ImageSets/Main'
source_image_root = '/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/JPEGImages'
target_image_root = '/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/JPEGImages_class_split'

# import pdb;pdb.set_trace()
for file in tqdm(os.listdir(root)):
    if file.endswith('_val.txt'):
        class_name = file.split('_')[0]
        real_target_image_root = os.path.join(target_image_root,class_name)
        if not os.path.exists(real_target_image_root):
            os.makedirs(real_target_image_root)
        file_name = os.path.join(root,file)
        with open(file_name) as f:
            metas = f.readlines()
        for line in metas:
            # try:
            path = line.strip().split(' ')[0]
            tag = line.strip().split(' ')[-1]
            if tag == '1':
                source_path = os.path.join(source_image_root,path+'.jpg')
                shutil.copy(source_path,real_target_image_root)
            # except:
            #     import pdb;pdb.set_trace()
