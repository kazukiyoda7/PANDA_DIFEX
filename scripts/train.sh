python3 panda_difex.py \
    --dataset cifar10 \
    --label 0 \
    --epochs 30 \
    --batch_size 32 \
    --resnet_type 18 \
    --output_dir train_results \
    --seed 42 \
    --severity 1 \
    --interval 10 \
    --lr_fourier 1e-4 \
    --alpha 0 \
    --beta 0 \
    --theta 0 \
    --epochs_fourier 1\
    --data_root ~/data \
    --domain 'clean-gaussian_noise'\