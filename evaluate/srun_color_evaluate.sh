
#!/bin/sh

mkdir -p logs
currenttime=`date "+%Y%m%d_%H%M%S"`

for metasplit in 7 #1 2 3 #4 5 6 7
do
    export MASTER_PORT=$((12000 + $RANDOM % 20000))
    srun -p ntu --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 -w SG-IDC2-10-51-5-53 \
    python evaluate_colorization.py --model mae_vit_large_patch16 --output_dir output_color_images/output_trn_${metasplit} --ckpt ../weights/checkpoint-1000.pth --data_path /mnt/lustre/yhzhang/data/imagenet --meta_split ${metasplit} \
    2>&1 | tee -a logs/${metasplit}-${currenttime}.log > /dev/null &
    echo -e "\033[32m[ Please check log: \"logs/${metasplit}-${currenttime}.log\" for details. ]\033[0m"
done

