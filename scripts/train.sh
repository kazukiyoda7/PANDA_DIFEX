python3 panda.py \
    --dataset cifar10 \
    --label 0 \
    --epochs 30 \
    --resnet_type 18 \
    --output_dir train_results \
    --seed 42 \
    --severity 1 \
    --interval 10 \
    --lr_fourier 1e-4 \
    --alpha 1 \
    --beta 1\
    --theta 1\
    --epochs_fourier 5\
    --data_root ~/data
    # --noise snow