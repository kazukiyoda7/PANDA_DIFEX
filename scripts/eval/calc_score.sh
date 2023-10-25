method=$1
model_path=$2
feature_path=$3
eval_domain=$4
id_class=$5

python3 eval/calc_score.py \
    --dataset cifar10 \
    --label $id_class \
    --seed 42 \
    --severity 1 \
    --eval_domain $eval_domain \
    --model_path $model_path \
    --feature_path $feature_path \
    --output_dir eval_results/$method \
    --data_root ~/data \
    --method $method \

