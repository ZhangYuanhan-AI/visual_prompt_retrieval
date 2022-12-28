
#!/bin/sh

mkdir -p logs
currenttime=`date "+%Y%m%d_%H%M%S"`

export MASTER_PORT=$((12000 + $RANDOM % 20000))
srun -p ntu --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 -w SG-IDC2-10-51-5-46 \
python evaluate_colorization.py --model mae_vit_large_patch16 --output_dir output_color_images/output_unsup --ckpt ../weights/checkpoint-1000.pth --data_path /mnt/lustre/yhzhang/data/imagenet  \
2>&1 | tee -a logs/${currenttime}.log > /dev/null &
echo -e "\033[32m[ Please check log: \"logs/${currenttime}.log\" for details. ]\033[0m"

