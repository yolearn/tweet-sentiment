import tokenizers
import os 
import pandas as pd
import torch

TRAIN_FILE = "../input/train.csv"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCH = 5 
LR = 3e-5
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