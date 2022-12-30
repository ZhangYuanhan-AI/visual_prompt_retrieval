import json 
with open('evaluate/output_random/log.txt') as f:
    metas = f.readlines()

iou_list = []
for idx, i in enumerate(metas[1:]):
    # import pdb;pdb.set_trace()
    try:
        a = float(eval(i.split('\t')[1])['iou'])
        iou_list.append(a)
    except:
        import pdb;pdb.set_trace()
        print(idx,i)
print(sum(iou_list)/len(iou_list))

    