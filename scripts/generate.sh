#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/configs/model_glm_130b.sh"

SEED=1234
MAX_OUTPUT_LENGTH=256
MIN_GEN_LENGTH=0
# BeamSearchStrategy args
NUM_BEAMS=4
LENGTH_PENALTY=1.0
NO_REPEAT_NGRAM=3
# BaseStrategy args
TEMP=0.9
TOPK=1
TOPP=0

ARGS="${main_dir}/generate.py \
       --seed $SEED \
       --mode inference \
       --sampling-strategy BeamSearchStrategy \
       --out-seq-length $MAX_OUTPUT_LENGTH \
       --min-gen-length $MIN_GEN_LENGTH \
       --num-beams $NUM_BEAMS \
       --length-penalty $LENGTH_PENALTY \
       --no-repeat-ngram-size $NO_REPEAT_NGRAM \
       --temperature $TEMP \
       --top_k $TOPK \
       --top_p $TOPP \
       --output-path samples \
       $MODEL_ARGS \
       $*"

run_cmd="torchrun --nproc_per_node $MP_SIZE ${ARGS}"
eval ${run_cmd}
