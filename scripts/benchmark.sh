#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/configs/model_glm_130b.sh"

ARGS="${main_dir}/benchmark.py \
       --mode inference \
       $MODEL_ARGS"

TIMESTAMP=$(date +'%Y.%m.%d-%H:%M:%S')
EXP_NAME=${TIMESTAMP}

mkdir -p logs

run_cmd="torchrun --nproc_per_node $MP_SIZE ${ARGS}"
echo $run_cmd
eval ${run_cmd} 2>&1 | tee logs/${EXP_NAME}.log
