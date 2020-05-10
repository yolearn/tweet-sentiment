import tokenizers
import os 
import pandas as pd
import torch
import transformers

#CUDA_VISIBLE_DEVICES=1 
TRAIN_FILE = "../input/train.csv"
SEED = 42
NFOLDS = 3
SHUFFLE = True
SPLIT_TYPE = 'kfold'
MAX_LEN = 128
BATCH_SIZE = 8
EPOCH = 1
LR = 5e-5
THRESHOLD = 0.3
MODEL_PATH = '../model/model.pth'


BERT_PATH = 'bert-base-uncased'
ROBERT_PATH = 'roberta-base'

#download bert tokenizer file
# bert_tokenizer = transformers.BertTokenizer.from_pretrained(BERT_PATH)
# bert_tokenizer.save_vocabulary("../input/")
# #download robert tokenizer file
# robert_tokenizer = transformers.RobertaTokenizer.from_pretrained(ROBERT_PATH)
# robert_tokenizer.save_vocabulary("../input/")

BERT_TOKENIZER = tokenizers.BertWordPieceTokenizer(
    "../input/vocab.txt",
    lowercase=True
)

ROBERT_TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file="../input/vocab.json",
    merges_file="../input/merges.txt",
    lowercase=True
)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else :
    DEVICE = torch.device('cpu')

if __name__ == "__main__":
    print(os.path.join(BERT_PATH, "vocab.txt"))
    print(pd.read_csv(TRAIN_FILE).keys())
    print(DEVICE)
    
    d = ROBERT_TOKENIZER.encode('fucking dube charlie')
    #print(d)
    print(d.ids)
    # print(d.type_ids)
    print(d.tokens)
    # print(d.attention_mask)
    # print(d.special_tokens_mask)
    print(d.offsets)

    d = BERT_TOKENIZER.encode('fucking dube charlie')
    print(d)