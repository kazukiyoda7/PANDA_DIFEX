method=$1

python3 eval/eval_semantic_shift.py \
    --dataset cifar10 \
    --label 0 \
    --seed 42 \
    --severity 1 \
    --eval_domain all \
    --output_dir eval_results/$method \
    --data_root ~/data \