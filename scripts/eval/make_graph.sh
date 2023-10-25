method=$1
eval_domain=$2
id_class=$3

python3 eval/make_score_graph.py \
    --seed 42 \
    --eval_domain $eval_domain \
    --input_dir eval_results/$method/score \
    --output_dir eval_results/$method \
    --id_class $id_class\