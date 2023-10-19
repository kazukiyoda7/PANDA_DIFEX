#!/bin/bash

method=$1
model_path=$2
feature_path=$3
shift 3

eval_domain="clean-gaussian_noise"
id_class=0

# 関数定義
calc_score() {
    bash scripts/eval/calc_score.sh $method $model_path $feature_path $eval_domain $id_class
}

eval_semantic_shift() {
    bash scripts/eval/eval_semantic_shift.sh $method $eval_domain $id_class
    bash scripts/eval/make_graph.sh $method $eval_domain $id_class
}

make_graph_ood() {
    bash scripts/eval/make_graph_ood.sh $method $eval_domain $id_class
}

for operation in "$@"; do
    case $operation in
        calc_score)
            calc_score
            ;;
        eval_semantic_shift)
            eval_semantic_shift
            ;;
        make_graph_ood)
            make_graph_ood
            ;;
        *)
            echo "Invalid operation: $operation. Choose from [calc_score, eval_semantic_shift, make_graph_ood]"
            ;;
    esac
done
