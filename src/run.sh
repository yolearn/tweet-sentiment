#roberta squad
CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --MODEL_TYPE roberta \
    --MODEL_VERSION roberta-base-squad2 \
    --MODEL_NAME roberta_base_squad2_pre_pro \
    --NFOLDS 10 \
    --EPOCH 10

    #--PRE_CLEAN y

#roberta origin
# CUDA_VISIBLE_DEVICES=1 python3 train.py \
#     --MODEL_TYPE roberta \
#     --MODEL_VERSION roberta-base \
#     --MODEL_NAME roberta_base_pre_pro \
#     #--PRE_CLEAN y
#roberta origin
#CUDA_VISIBLE_DEVICES=1 python3 train.py --MODEL_TYPE bert --MODEL_VERSION bert-base-uncased --MODEL_NAME bert_base_baseline
