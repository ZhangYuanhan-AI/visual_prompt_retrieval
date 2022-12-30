import json

with open ('/mnt/lustre/share/yhzhang/pascal-5i/VOC2012/JPEGImages_class_split/domains.json') as f:
    cluster = json.load(f)


cluster_dict = {}
original_label2real_label = {}
counter = 1
for idx, class_name in enumerate(cluster.keys()):
    for image in cluster[class_name]:
        cluster_label = cluster[class_name][image]
        real_cluster_label = str(idx)+str(int(cluster_label))
        if real_cluster_label not in original_label2real_label:
            original_label2real_label[real_cluster_label] = counter
            counter += 1
        real_image_name = image.split('/')[1][:-4]
        # import pdb;pdb.set_trace()
        cluster_dict[real_image_name] = str(original_label2real_label[real_cluster_label])+'\n'


for foldid in ["0","1","2","3"]:
    with open('/mnt/lustre/yhzhang/visual_prompting/evaluate/splits/pascal/val/fold'+foldid+'.txt') as f:
        metas = f.readlines()
    # import pdb;pdb.set_trace()
    with open('/mnt/lustre/yhzhang/visual_prompting/evaluate/splits/pascal/val/fold_cluster'+foldid+'.txt','w') as f:#, open('/mnt/lustre/yhzhang/visual_prompting/evaluate/splits/pascal/val/fold_class'+foldid+'.txt','w') as f1:
        for i in metas:
            name = i.split('__')[0]
            if name in cluster_dict:
                f.write(i.strip()+'__'+cluster_dict[name])
                # f1.write(i)
            else:
                pass
                # import pdb;pdb.set_trace()


