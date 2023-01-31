
#!/bin/sh

mkdir -p logs
currenttime=`date "+%Y%m%d_%H%M%S"`

feature_name=$1
output_name=$2


for folderid in 0 1 2 3 #2
do
    for seed in 0 #1 2 #3
    do
        export MASTER_PORT=$((12000 + $RANDOM % 20000))
        srun -p ntu --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 -w SG-IDC2-10-51-5-46 \
        python evaluate_segmentation.py --model mae_vit_large_patch16 --base_dir ~/data/pascal-5i/ --feature_name ${feature_name} --output_dir output_seg_images/${output_name}_folder${folderid}_seed${seed} --ckpt ../weights/checkpoint-1000.pth --fold ${folderid} --seed ${seed} \
        2>&1 | tee -a logs/${folderid}-${seed}-${currenttime}.log > /dev/null &
        echo -e "\033[32m[ Please check log: \"logs/${folderid}-${seed}-${currenttime}.log\" for details. ]\033[0m"
    done
done