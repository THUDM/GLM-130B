MODEL_TYPE="glm-roberta-large"
CHECKPOINT_PATH="/zhangpai21/checkpoints/glm-large-en-blank"
MP_SIZE=1
MODEL_ARGS="--model-parallel-size ${MP_SIZE} \
            --vocab 50304 \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 513 \
            --tokenizer-type glm_GPT2BPETokenizer \
            --tokenizer-model-type roberta \
            --task-mask \
            --load ${CHECKPOINT_PATH}"