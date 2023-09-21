python3 eval_robustness.py \
    --dataset cifar10 \
    --label 0 \
    --seed 42 \
    --severity 1 \
    --eval_domain clean-gaussian_noise-fog \
    --model_path trained_model/resnet18/clean+snow/model_30.pth \
    --feature_path trained_model/resnet18/clean+snow/train_feature_30.npy