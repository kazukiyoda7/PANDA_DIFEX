method=$1
eval_domain=$2
id_class=$3

python3 eval/eval_semantic_shift.py \
    --dataset cifar10 \
    --label $id_class \
    --seed 42 \
    --severity 1 \
    --eval_domain $eval_domain \
    --output_dir eval_results/$method \
    --data_root ~/data \