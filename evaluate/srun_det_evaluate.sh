#!/bin/sh

mkdir -p logs
currenttime=`date "+%Y%m%d_%H%M%S"`




for seed in 0 #1 2 3
do
    export MASTER_PORT=$((12000 + $RANDOM % 20000))
    srun -p ntu --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 -w SG-IDC2-10-51-5-46 \
    python evaluate_segmentation.py --task detection --model mae_vit_large_patch16 --base_dir ~/data/pascal-5i/ --feature_name features_supcon-in1k-pretrain_det --output_dir output_det_images/output_supcon-in1k-pretrain_val_${seed} --ckpt ../weights/checkpoint-1000.pth --seed ${seed} --dataset_type pascal_det \
    2>&1 | tee -a logs/${seed}-${currenttime}.log > /dev/null &
    echo -e "\033[32m[ Please check log: \"logs/${seed}-${currenttime}.log\" for details. ]\033[0m"
done
