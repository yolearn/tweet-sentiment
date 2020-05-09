import tokenizers
import os 
import pandas as pd
import torch
#CUDA_VISIBLE_DEVICES=1 
TRAIN_FILE = "../input/train.csv"
SEED = 42
NFOLDS = 3
SHUFFLE = True
SPLIT_TYPE = 'kfold'
MAX_LEN = 128
BATCH_SIZE = 32
EPOCH = 1
LR = 3e-5
THRESHOLD = 0.3
MODEL_PATH = '../model/model.pth'

BERT_PATH = 'bert-base-uncased'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    "../input/vocab.txt",
    lowercase=True,
)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else :
    DEVICE = torch.device('cpu')

if __name__ == "__main__":
    print(os.path.join(BERT_PATH, "vocab.txt"))
    print(pd.read_csv(TRAIN_FILE).keys())
    print(DEVICE)