python3 panda_disentangle.py \
    --dataset cifar10 \
    --label 0 \
    --epochs 30 \
    --resnet_type 18 \
    --output_dir train_results \
    --seed 42 \
    --severity 1 \
    --interval 10 \
    --data_root ~/data\
    --domain clean-gaussian_noise \
    --batch_size 16 \
    --alpha 1 \
    --beta 1 \
    --lr 1e-3 \
    --optuna \
    --n_trials 30 \
    # --ewc 