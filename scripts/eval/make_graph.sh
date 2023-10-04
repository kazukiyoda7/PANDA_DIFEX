method=$1

python3 eval/make_score_graph.py \
    --seed 42 \
    --input_dir eval_results/$method/score \
    --output_dir eval_results/$method \
    --id_class 0\