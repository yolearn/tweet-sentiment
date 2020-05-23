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

 
MODEL_NAME = 'roberta-base-squad2_baseline'
if not os.path.exists(f'../model/{MODEL_NAME}'):
    os.makedirs(f'../model/{MODEL_NAME}')

SAVE_MODEL = True
TRN_NUM = None
TRAIN_FILE = "../input/train.csv"
SEED = 42
NFOLDS = 5
SHUFFLE = True
SPLIT_TYPE = 'kfold'
#SPLIT_TYPE = 'pure_split'
MAX_LEN =  128
BATCH_SIZE = 32
EPOCH = 5
LR = 3e-5
DROPOUT_RATE = 0.2
PATIENCE = 3


# MODEL_TYPE = 'albert'
MODEL_TYPE = "roberta"
#MODEL_TYPE = "bert"
BUCKET_NAME = "kaggletweet"


# = '../model/model.pth'

if MODEL_TYPE == 'bert':
    BERT_TOKENIZER = tokenizers.BertWordPieceTokenizer(
        "../input/vocab.txt",
        lowercase=True,
    )
    TOKENIZER = BERT_TOKENIZER
    #bert-base-uncased
    #bert-large-uncased
    #bert-large-uncased-whole-word-masking-finetuned-squad
    MODEL_PATH = 'bert-large-uncased'

elif MODEL_TYPE == 'roberta':
    ROBERT_TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab_file="../input/roberta-base-squad2/vocab.json",
        merges_file="../input/roberta-base-squad2/merges.txt",
        lowercase=True,
        add_prefix_space=True
    )
    TOKENIZER = ROBERT_TOKENIZER
    #roberta-base
    #roberta-large
    #roberta-base-squad2
    MODEL_PATH = '../input/roberta-base-squad2'
    MODEL_CONF = '../input/roberta-base-squad2/config.json'

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
    pass



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