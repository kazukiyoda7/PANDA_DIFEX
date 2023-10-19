method=$1
eval_domain=$2
id_class=$3

for ood_class in {0..9}; do
    if [ $ood_class -ne $id_class ]; then
        python3 eval/make_score_graph_ood.py \
            --seed 42 \
            --input_dir eval_results/$method/score \
            --output_dir eval_results/$method \
            --eval_domain $eval_domain \
            --id_class $id_class \
            --ood_class $ood_class

    fi
done

