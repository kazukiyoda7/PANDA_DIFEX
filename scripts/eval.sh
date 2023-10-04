#!/bin/bash

method=$1
model_path=$2
feature_path=$3
shift 3

# 関数定義
calc_score() {
    bash scripts/eval/calc_score.sh $method $model_path $feature_path
}

eval_semantic_shift() {
    bash scripts/eval/eval_semantic_shift.sh $method
    bash scripts/eval/make_graph.sh $method
}

make_graph2() {
    bash scripts/eval/make_graph2.sh $method
}

for operation in "$@"; do
    case $operation in
        calc_score)
            calc_score
            ;;
        eval_semantic_shift)
            eval_semantic_shift
            ;;
        make_graph2)
            make_graph2
            ;;
        *)
            echo "Invalid operation: $operation. Choose from [calc_score, eval_semantic_shift, make_graph2]"
            ;;
    esac
done
