python3 panda_disentangle.py \
    --dataset cifar10 \
    --label 0 \
    --epochs 30 \
    --resnet_type 18 \
    --output_dir train_results \
    --seed 42 \
    --severity 5 \
    --interval 10 \
    --data_root ~/data\
    --domain clean-fog \
    --batch_size 32 \
    --alpha 1 \
    --beta 1 \
    --panda \
    --disentangle
    # --optuna \
    # --n_trials 30 \
    # --ewc 