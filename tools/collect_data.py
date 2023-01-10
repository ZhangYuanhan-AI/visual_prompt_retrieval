import statistics 
import sys


output_list = [
    'output_domain-shot'
]
root = '/mnt/lustre/yhzhang/visual_prompting/evaluate/output_seg_images'
result_dict = {}
for cur_output in output_list:
    result_dict[cur_output] = {}
    for folderid in [0, 1 ,2 ,3]:
        result_dict[cur_output][folderid] = []
        for seed in [0, 1, 2]:
            meta_file = f"{root}/{cur_output}_{folderid}_{seed}/log.txt"
            try:
                with open(meta_file) as f:
                    metas = f.readlines()
                # import pdb;pdb.set_trace()
                result = eval(metas[-1].split('\t')[1])['iou']
                result_dict[cur_output][folderid].append(result)
            except:
                print(f"{meta_file} is not exist")
                sys.stdout.flush()
                del result_dict[cur_output][folderid]
                continue

# import pdb;pdb.set_trace()
for cur_output in output_list:
    for folderid in [0, 1 ,2 ,3]: 
        if folderid in result_dict[cur_output]:
            data = result_dict[cur_output][folderid]
            # import pdb;pdb.set_trace()
            print(f"Mean of the {cur_output}_{folderid} is {statistics.mean(data)}") 
            sys.stdout.flush()
            # print(f"Std of the {cur_output}_{folderid} is {statistics.stdev(data)}")
            # sys.stdout.flush()



