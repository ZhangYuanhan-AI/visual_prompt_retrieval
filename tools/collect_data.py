import statistics 
import sys


output_list = [
    'output_rn18_val',
    'output_supcon_bsz64_val',
    'output_supcon-in1k-pretrain_val',
    'output_supcon-vit-laion2b-freeze-encoder_val',
    'output_vit_val',
    'output_vit-eva_val',
    'output_vit-in21k-ft-in1k_val'


]
root = '/mnt/lustre/yhzhang/visual_prompting/evaluate/output_color_images'
result_dict = {}
for cur_output in output_list:
    result_dict[cur_output] = []
    for seed in [0, 1, 2]:
        meta_file = f"{root}/{cur_output}_{seed}/log.txt"
        try:
            with open(meta_file) as f:
                metas = f.readlines()
            # import pdb;pdb.set_trace()
            result = eval(metas[-1].split('\t')[1])['mse']
            result_dict[cur_output].append(result)
        except:
            print(f"{meta_file} is not exist")
            sys.stdout.flush()
            del result_dict[cur_output]
            continue

# import pdb;pdb.set_trace()
for cur_output in output_list:
    data = result_dict[cur_output]
    # import pdb;pdb.set_trace()
    print(f"Mean of the {cur_output} is {statistics.mean(data)}") 
    sys.stdout.flush()
    # print(f"Std of the {cur_output}_{folderid} is {statistics.stdev(data)}")
    # sys.stdout.flush()



