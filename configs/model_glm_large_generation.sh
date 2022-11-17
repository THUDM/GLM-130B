MODEL_TYPE="glm-large-generation"
CHECKPOINT_PATH="/zhangpai21/checkpoints/glm-large-en-generation"
MP_SIZE=1
MODEL_ARGS="--model-parallel-size ${MP_SIZE} \
            --vocab 30592 \
            --num-layers 24 \
            --hidden-size 1024 \
            --num-attention-heads 16 \
            --max-sequence-length 513 \
            --tokenizer-type glm_BertWordPieceTokenizer \
            --tokenizer-model-type bert-large-uncased \
            --task-mask \
            --load ${CHECKPOINT_PATH}"