import json
import os
 
inference_result_root = '/mnt/lustre/yhzhang/visual_prompting/evaluate/output_color_images/'

iou_dict = {}
for fold_id in range(8):
    cur_dir = os.path.join(inference_result_root, 'output_trn_{}/log.txt'.format(fold_id))
    
    with open(cur_dir) as f:
        metas = f.readlines()
    
    fold_n_metadata_path = '/mnt/lustre/yhzhang/data/imagenet/annotations/train_meta.list.num_shot_16.seed_0.{}'.format(fold_id)
    
    with open(fold_n_metadata_path, 'r') as f:
        fold_n_metadata = f.readlines()
        
    fold_n_metadata = [data.split(' ')[0].split('/')[1][:-5] for data in fold_n_metadata]
    
    with open('/mnt/lustre/yhzhang/data/imagenet/features_vit_train-shot16-seed0/top50-similarity.json') as f:
        images_top50 = json.load(f)
    
    for cur_line in metas[1:-1]:
        # import pdb;pdb.set_trace()
        img_id, sim_id, result = cur_line.split('\t')
        img_id, sim_id = int(img_id), int(sim_id)
        result = eval(result)
        mse = result['mse']
        image_name = fold_n_metadata[img_id]
        if image_name not in iou_dict:
            iou_dict[image_name] = {}
        iou_dict[image_name][images_top50[image_name][sim_id]] = mse
    
# delete the similarity of itself and then get the top5 and botton 5
for img_name in iou_dict:
    if img_name in iou_dict[img_name]:
        del iou_dict[img_name][img_name]
    sorted_iou = sorted(iou_dict[img_name].items(), key=lambda x:x[1])
    # import pdb;pdb.set_trace()
    sorted_iou_names = [x[0] for x in sorted_iou[:5]+sorted_iou[-5:]]
    iou_dict[img_name] = sorted_iou_names

save_dir = os.path.join(inference_result_root, 'contrastive.json')

# import pdb;pdb.set_trace()

with open(save_dir,'w') as f:
    json.dump(iou_dict, f)
 
 
 
 
 

