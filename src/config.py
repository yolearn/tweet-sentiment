import tokenizers
import os 
import pandas as pd

TRAIN_FILE = "../input/train.csv"
MAX_LEN = 100
BATCH_SIZE = 32
EPOCH = 5 

BERT_PATH = "../input/"
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, "vocab.txt"),
    lowercase=True,

)

if __name__ == "__main__":
    print(os.path.join(BERT_PATH, "vocab.txt"))
    print(pd.read_csv(TRAIN_FILE).keys())