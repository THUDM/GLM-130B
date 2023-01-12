MODEL_TYPE="glm-6b"
CHECKPOINT_PATH="<your checkpoint path>"
MP_SIZE=1
MODEL_ARGS="--model-parallel-size ${MP_SIZE} \
            --num-layers 28 \
            --hidden-size 4096 \
            --inner-hidden-size 16384 \
            --vocab-size 150528 \
            --num-attention-heads 32 \
            --max-sequence-length 2048 \
            --tokenizer-type icetk-glm-130B \
            --layernorm-order post \
            --position-encoding-2d \
            --no-glu \
            --load ${CHECKPOINT_PATH} \
            --skip-init \
            --fp16"
