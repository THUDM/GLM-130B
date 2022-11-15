MODEL_TYPE="glm-2b"
CHECKPOINT_PATH="/zhangpai21/checkpoints/glm-2b-sat"
MP_SIZE=1
MODEL_ARGS="--model-parallel-size ${MP_SIZE} \
            --vocab 50304 \
            --num-layers 36 \
            --hidden-size 2048 \
            --num-attention-heads 32 \
            --max-sequence-length 1025 \
            --tokenizer-type glm_GPT2BPETokenizer \
            --tokenizer-model-type gpt2 \
            --task-mask \
            --load ${CHECKPOINT_PATH}"