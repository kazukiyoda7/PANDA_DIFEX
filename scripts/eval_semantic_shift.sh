$method="difex-single" 

python3 eval/eval_semantic_shift.py \
    --dataset cifar10 \
    --label 0 \
    --seed 42 \
    --severity 0 \
    --eval_domain all \
    --model_path /home/yoda/workspace_dg/path/resnet18+gaussian/0/model/model_30.pth \
    --feature_path /home/yoda/workspace_dg/path/resnet18+gaussian/0/feature/train_feature_30.npy \
    --output_dir eval_results/$method \
    --data_root ~/data \