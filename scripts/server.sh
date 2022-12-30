#!/bin/bash

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

source "${main_dir}/configs/model_glm_130b.sh"

ARGS="${main_dir}/server.py \
       --mode inference \
       $MODEL_ARGS \
       $*"

run_cmd="torchrun --nproc_per_node $MP_SIZE ${ARGS}"
eval ${run_cmd}