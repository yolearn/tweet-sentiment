import tokenizers
import os 
import pandas as pd
import torch
import transformers
import sentencepiece_pb2
import sentencepiece as spm
import os
import sentencepiece_pb2

class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
    
    def encode(self, sentence):
        spt = sentencepiece_pb2.SentencePieceText()
        spt.ParseFromString(self.sp.encode_as_serialized_proto(sentence))
        offsets = []
        tokens = []
        for piece in spt.pieces:
            tokens.append(piece.id)
            offsets.append((piece.begin, piece.end))
        return tokens, offsets

#from train import args
#from train import args
#CUDA_VISIBLE_DEVICES=1 

# TRAIN_FILE = args['TRAIN_FILE']
# NFOLDS = args['nfolds']
# SHUFFLE = True
# SPLIT_TYPE = args['split_type']
# MAX_LEN = args['max_len']
# BATCH_SIZE = args['batch_size']
# EPOCH = args['epoch']
# LR = args['lr']
# DROPOUT_RATE = args['dropout_rate']
# PATIENCE = args['patience']

TRAIN_FILE = "../input/train.csv"
SEED = 42
NFOLDS = 5
SHUFFLE = True
SPLIT_TYPE = 'kfold'
#SPLIT_TYPE = 'pure_split'
MAX_LEN = 95
BATCH_SIZE = 32
EPOCH = 10
LR = 3e-5
DROPOUT_RATE = 0.2
PATIENCE = 3

MODEL_TYPE = 'roberta'
BUCKET_NAME = "kaggletweet"

# = '../model/model.pth'

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
        lowercase=False
    )
    TOKENIZER = ROBERT_TOKENIZER
    #roberta-base
    #roberta-large
    MODEL_PATH = 'roberta-base'

elif MODEL_TYPE == 'albert':
    MODEL_PATH = 'albert-base-v2'
    #ALBERT_TOKENIZER = transformers.ALBERT_TOKENIZER(MODEL_PATH)
    Seq_tokenizer = SentencePieceTokenizer('../input/spiece.model')
    TOKENIZER =  Seq_tokenizer  



if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else :
    DEVICE = torch.device('cpu')

if __name__ == "__main__":
    # print(os.path.join(BERT_PATH, "vocab.txt"))
    # print(pd.read_csv(TRAIN_FILE).keys())
    # print(DEVICE)
    
    # #download bert tokenizer file
    # bert_tokenizer = transformers.BertTokenizer.from_pretrained(BERT_PATH)
    # bert_tokenizer.save_vocabulary("../input/")
    # # #download robert tokenizer file
    # robert_tokenizer = transformers.RobertaTokenizer.from_pretrained(ROBERT_PATH)
    # robert_tokenizer.save_vocabulary("../input/")

    out = TOKENIZER.encode("87 charlie don't understand anything")
    print(out.offsets)
    print(out.tokens)
    text = "87 charlie don't understand anything"
    offsets = out.offsets
    start_idx = 3
    end_idx = 5

    fin = ''
    for i in range(start_idx+1, end_idx+2):
        fin+=text[offsets[i][0]:offsets[i][1]]
    
    print(fin)


