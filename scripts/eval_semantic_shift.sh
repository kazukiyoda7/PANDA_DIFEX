python3 eval_semantic_shift.py \
    --dataset cifar10 \
    --label 0 \
    --seed 42 \
    --severity 0 \
    --eval_domain all \
    --output_dir result_csv/resnet18-gaussian \
    --model_path /home/yoda/workspace_dg/path/resnet18+gaussian/0/model/model_30.pth \
    --feature_path /home/yoda/workspace_dg/path/resnet18+gaussian/0/feature/train_feature_30.npy \
    --score_save_dir result_score/resnet18-gaussian