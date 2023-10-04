method=$1

python3 eval/make_score_graph2.py \
    --seed 42 \
    --input_dir eval_results/$method/score \
    --output_dir eval_results/$method \
    --id_class 0\
    --ood_class 1
