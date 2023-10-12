python3 panda_vanilla.py \
    --dataset cifar10 \
    --label 0 \
    --epochs 30 \
    --resnet_type 18 \
    --output_dir train_results \
    --seed 42 \
    --severity 5 \
    --interval 10 \
    --data_root ~/data\
    --domain clean-gaussian_noise \
    --batch_size 32 \
    # --ewc 