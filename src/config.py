import tokenizers
import os 
import pandas as pd
import torch
import transformers

#CUDA_VISIBLE_DEVICES=1 
TRAIN_FILE = "../input/train.csv"
SEED = 42
NFOLDS = 5
SHUFFLE = True
SPLIT_TYPE = 'kfold'
MAX_LEN = 192
BATCH_SIZE = 32
EPOCH = 5
LR = 3e-5
THRESHOLD = 0.3
MODEL_TYPE = 'roberta'
#MODEL_PATH = '../model/model.pth'

if MODEL_TYPE == 'bert':
    BERT_TOKENIZER = tokenizers.BertWordPieceTokenizer(
        "../input/vocab.txt",
        lowercase=True
    )
    TOKENIZER = BERT_TOKENIZER
    MODEL_PATH = 'bert-base-uncased'

elif MODEL_TYPE == 'roberta':
    ROBERT_TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab_file="../input/vocab.json",
        merges_file="../input/merges.txt",
        lowercase=True
    )
    TOKENIZER = ROBERT_TOKENIZER
    MODEL_PATH = 'roberta-base'


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else :
    DEVICE = torch.device('cpu')

if __name__ == "__main__":
    print(os.path.join(BERT_PATH, "vocab.txt"))
    print(pd.read_csv(TRAIN_FILE).keys())
    print(DEVICE)
    
    #download bert tokenizer file
    bert_tokenizer = transformers.BertTokenizer.from_pretrained(BERT_PATH)
    bert_tokenizer.save_vocabulary("../input/")
    # #download robert tokenizer file
    robert_tokenizer = transformers.RobertaTokenizer.from_pretrained(ROBERT_PATH)
    robert_tokenizer.save_vocabulary("../input/")