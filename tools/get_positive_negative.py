import json
import os


inference_result_root = '/mnt/lustre/yhzhang/visual_prompting/evaluate/output_seg_images/'

feature_name = sys.argv[1]
output_name = sys.argv[2]

for fold_id in range(4):
    cur_dir = os.path.join(inference_result_root, f'{output_name}_{fold_id}_0/log.txt')
    
    with open(cur_dir) as f:
        metas = f.readlines()
    
    fold_n_metadata_path = f'/mnt/lustre/yhzhang/visual_prompting/evaluate/splits/pascal/trn/fold{fold_id}.txt'

    with open(fold_n_metadata_path, 'r') as f:
        fold_n_metadata = f.read().split('\n')[:-1]
        
    # import pdb;pdb.set_trace()
    fold_n_metadata = [data.split('__')[0] for data in fold_n_metadata]

    with open(f'/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/{feature_name}/folder{fold_id}_top50-similarity.json') as f:
        images_top50 = json.load(f)

    iou_dict = {}
    for cur_line in metas[1:-1]:
        # import pdb;pdb.set_trace()
        img_id, sim_id, result = cur_line.split('\t')
        img_id, sim_id = int(img_id), int(sim_id)
        result = eval(result)
        iou = result['iou']
        image_name = fold_n_metadata[img_id]
        if image_name not in iou_dict:
            iou_dict[image_name] = {}
        iou_dict[image_name][images_top50[image_name][sim_id]] = iou
    
    # delete the similarity of itself and then get the top5 and botton 5
    # import pdb;pdb.set_trace()
    for img_name in iou_dict:
        if img_name in iou_dict[img_name]:
            del iou_dict[img_name][img_name]
        import pdb;pdb.set_trace()
        sorted_iou = sorted(iou_dict[img_name].items(), key=lambda x:x[1], reverse=True)
        sorted_iou_names = [x[0] for x in sorted_iou[:5]+sorted_iou[-5:]]
        iou_dict[img_name] = sorted_iou_names
    
    save_dir = os.path.join(inference_result_root, f'{output_name}_{fold_id}_0/contrastive.json')
    with open(save_dir,'w') as f:
        json.dump(iou_dict, f)




