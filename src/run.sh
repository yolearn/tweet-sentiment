#roberta squad
CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --MODEL_VERSION roberta-base-squad2 \
    --MODEL_NAME 0614_1 \
    --BATCH_SIZE 32 \
    --MAX_LEN 95 \
    --EPOCH 2



# parser.add_argument('--EMBEDDING_SIZE', default=768, type=int)
# parser.add_argument('--CNN_KERNEL_SZIE', default=1, type=int)
# parser.add_argument('--CNN_OUTPUT_CHANNEL', default=1, type=int)
# parser.add_argument('--BATCH_SIZE', default=32, type=int)
# parser.add_argument('--MODEL_VERSION', default='roberta-base')
# parser.add_argument('--MAX_LEN', default=128, type=int)
# parser.add_argument('--EPOCH', default=3, type=int)
# parser.add_argument('--LR', default=3e-5, type=int)
# parser.add_argument('--PATIENCE', default=1, type=int)

#roberta origin
# CUDA_VISIBLE_DEVICES=1 python3 train.py \
#     --MODEL_TYPE roberta \
#     --MODEL_VERSION roberta-base \
#     --MODEL_NAME roberta_base_pre_pro \
#     #--PRE_CLEAN y
#roberta origin
#CUDA_VISIBLE_DEVICES=1 python3 train.py --MODEL_TYPE bert --MODEL_VERSION bert-base-uncased --MODEL_NAME bert_base_baseline
