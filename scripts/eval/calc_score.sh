method=$1
model_path=$2
feature_path=$3

python3 eval/calc_score.py \
    --dataset cifar10 \
    --label 0 \
    --seed 42 \
    --severity 1 \
    --eval_domain all \
    --model_path $model_path \
    --feature_path $feature_path \
    --output_dir eval_results/$method \
    --data_root ~/data \
    --method $method

