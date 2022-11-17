MODEL_TYPE="glm-large-generation"
CHECKPOINT_PATH="/zhangpai21/checkpoints/glm-base-en-blank"
MP_SIZE=1
MODEL_ARGS="--model-parallel-size ${MP_SIZE} \
            --vocab 30592 \
            --num-layers 12 \
            --hidden-size 768 \
            --num-attention-heads 12 \
            --max-sequence-length 513 \
            --tokenizer-type glm_BertWordPieceTokenizer \
            --tokenizer-model-type bert-base-uncased \
            --load ${CHECKPOINT_PATH}"