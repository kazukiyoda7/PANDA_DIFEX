method="vanilla+gaussian_noise+impulse_noise" 

python3 eval/eval_semantic_shift.py \
    --dataset cifar10 \
    --label 0 \
    --seed 42 \
    --severity 0 \
    --eval_domain all \
    --model_path /home/yoda/workspace_dg/PANDA_DIFEX/train_results/2023-9-30/14-6-1/model/model_30.pth \
    --feature_path /home/yoda/workspace_dg/PANDA_DIFEX/train_results/2023-9-30/14-6-1/feature/train_feature_30.npy \
    --output_dir eval_results/$method \
    --data_root ~/data \


python3 eval/make_score_graph.py \
    --seed 42 \
    --input_dir eval_results/$method/score \
    --output_dir eval_results/$method \
    --id_class 0\