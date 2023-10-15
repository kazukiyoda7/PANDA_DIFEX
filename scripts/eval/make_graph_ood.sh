method=$1
id_class=0

for ood_class in {0..9}; do
    if [ $ood_class -ne $id_class ]; then
        python3 eval/make_score_graph_ood.py \
            --seed 42 \
            --input_dir eval_results/$method/score \
            --output_dir eval_results/$method \
            --id_class $id_class \
            --ood_class $ood_class

    fi
done

