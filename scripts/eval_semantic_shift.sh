python3 eval_semantic_shift.py \
    --dataset cifar10 \
    --label 0 \
    --seed 42 \
    --severity 0 \
    --eval_domain all \
    --output_dir result_csv/resnet18 \
    --model_path /home/yoda/workspace_dg/PANDA_DIFEX/results/2023-9-21/14-16-8/model/model_15.pth \
    --feature_path /home/yoda/workspace_dg/PANDA_DIFEX/results/2023-9-21/14-16-8/feature/train_feature_15.npy \
    --score_save_dir result_score/resnet18