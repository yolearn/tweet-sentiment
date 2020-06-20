import time
import boto3
from botocore.exceptions import NoCredentialsError
from model import RobertUncaseQa
import os
import torch
import torch.nn as nn
import numpy as np
import re 
import string
import config
from aws_setting import ACCESS_KEY, SECRET_KEY

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)
    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False

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


class EarlyStopping():
    def __init__(self, path, patience=5):
        self.patience = patience
        self.counter = 0
        self.earlystop = False
        self.path = path
        self.cur = 0 
        self.max = 0
        self.model = 0
        self.epoch = 0
        self.pred1 = 0
        self.pred2 = 0

    def __call__(self, cur, model, epoch, pred1, pred2):
        self.cur = cur
        self.model = model

        if self.cur > self.max:
            self.max = self.cur
            self.pred1 = pred1
            self.pred2 = pred2
            self.epoch = epoch
            self.counter = 0
            self.save_model()

        else :
            self.counter+=1
            print(f'EarlyStopping counter : {self.counter} out of {self.patience}')
            
        if self.counter > self.patience:
            self.earlystop = True
            print(f'The best epoch is {self.epoch}')
            print(f'The best score is {self.max}')

    def save_model(self):
        #using parrallel will haveprefix "module."
        print('saving.....')
        try :
            torch.save(self.model.module.state_dict(), self.path)
        except:
            torch.save(self.model.state_dict(), self.path)
    

def cal_jaccard(fin_output_start, fin_output_end, fin_offset, fin_orig_sentiment, fin_orig_selected, fin_orig_text, rm_length):
    """
    calculate jaccard
    :fin_output_start -> (batch_size, max_len)
    :fin_output_end   -> (batch_size, max_len)
    :return: mean jaccard
    """
    neu_score = []
    pos_score = []
    neg_score = []
    jaccard_score = []
    for i in range(fin_output_start.shape[0]):
        output_start = fin_output_start[i]
        output_start = np.argmax(output_start)
        output_end = fin_output_end[i]
        output_end = np.argmax(output_end)
        offset = fin_offset[i]
        orig_sentiment = fin_orig_sentiment[i]
        orig_selected = fin_orig_selected[i]
        orig_text = fin_orig_text[i]
    
        if output_start > output_end:
            output_end = output_start
        
        output_string = ""
        for j in range(output_start, output_end+1):
            output_string += orig_text[offset[j][0]:offset[j][1]]
            if (j+1) < len(offset) and offset[j][1] < offset[j+1][0]:
               output_string += " "
        
        #output_string = post_process(output_string)
        output_string = output_string.strip()
        if orig_sentiment == 'neutral' or len(orig_text.split()) < rm_length:
            output_string = orig_text
            neu_score.append(jaccard(output_string, orig_selected))

        elif orig_sentiment == 'positive':
            pos_score.append(jaccard(output_string, orig_selected))

        else:
            neg_score.append(jaccard(output_string, orig_selected))
 

    # print('neutral score : ', sum(neu_score) / len(neu_score) if len(neu_score)!=0 else None)
    # print('positive score : ', sum(pos_score) / len(pos_score) if len(pos_score)!=0 else None)
    # print('negtive score : ', sum(neg_score) / len(neg_score) if len(neg_score)!=0 else None)

    return {
        'avg_score' : (sum(neu_score) + sum(pos_score) + sum(neg_score)) / (len(neu_score) + len(pos_score) + len(neg_score)),
        'neu_score' : sum(neu_score) / len(neu_score) if len(neu_score)!=0 else None,
        'pos_score' : sum(pos_score) / len(pos_score) if len(pos_score)!=0 else None,
        'neg_score' : sum(neg_score) / len(neg_score) if len(neg_score)!=0 else None
    }

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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

def cal_accucary(fin_output_sentiment, fin_targ_sentiment):
    correct = 0

    fin_output_sentiment = np.argmax(fin_output_sentiment, axis=1)
    for i in range(len(fin_output_sentiment)):
        if fin_output_sentiment[i] == fin_targ_sentiment[i]:
            correct+=1
    
    
    return correct / len(fin_output_sentiment)

def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def pre_process(df, length=4):
    #print(df.head())
    # df['text'] = df['text'].apply(lambda x:clean_text(x))
    # df['selected_text'] = df['selected_text'].apply(lambda x:clean_text(x))
    print(f'Removing length < {length} ....')
    df = df[df['sentiment'] != 'neutral']
    df['text_len'] = df['text'].apply(lambda x : len(x.split()))
    df = df[df['text_len'] > length-1]

    return df
    
def post_process(s):
    a = re.findall('[^A-Za-z0-9]',s)
    b = re.sub('[^A-Za-z0-9]+', '', s)

    try:
        if a.count('.')==3:
            text = b + '. ' + b + '..'
        elif a.count('!')==4:
            text = b + '! ' + b + '!! ' +  b + '!!!'
        else:
            text = s
        return text
    except:
        return text

def load_model(file_path):
    model1 = RobertUncaseQa(
                robert_path=config.MODEL_PATH, 
                conf=config.MODEL_CONF, 
                embedding_size=config.EMBEDDING_SIZE, 
                cnn_output_channel=config.CNN_OUTPUT_CHANNEL,
                kernel_width=config.CNN_KERNEL_WIDTH, 
                dropout_rate=config.DROPOUT_RATE
            ).to(config.DEVICE)

    model1.load_state_dict(torch.load(f'../model/{file_path}/fold1.pth'))

    model2 = RobertUncaseQa(
                robert_path=config.MODEL_PATH, 
                conf=config.MODEL_CONF, 
                embedding_size=config.EMBEDDING_SIZE, 
                cnn_output_channel=config.CNN_OUTPUT_CHANNEL,
                kernel_width=config.CNN_KERNEL_WIDTH, 
                dropout_rate=config.DROPOUT_RATE
            ).to(config.DEVICE)

    model2.load_state_dict(torch.load(f'../model/{file_path}/fold2.pth'))

    model3 = RobertUncaseQa(
                robert_path=config.MODEL_PATH, 
                conf=config.MODEL_CONF, 
                embedding_size=config.EMBEDDING_SIZE, 
                cnn_output_channel=config.CNN_OUTPUT_CHANNEL,
                kernel_width=config.CNN_KERNEL_WIDTH, 
                dropout_rate=config.DROPOUT_RATE
            ).to(config.DEVICE)

    model3.load_state_dict(torch.load(f'../model/{file_path}/fold3.pth'))

    model4 = RobertUncaseQa(
                robert_path=MODEL_PATH, 
                conf=config.MODEL_CONF, 
                embedding_size=args['EMBEDDING_SIZE'], 
                cnn_output_channel=args['CNN_OUTPUT_CHANNEL'],
                kernel_width=args['CNN_KERNEL_WIDTH'], 
                dropout_rate=args['DROPOUT_RATE']).to(args['DEVICE']
            )
    model4.load_state_dict(torch.load(f'../model/{file_path}/model_fold4.pth'))

    model5 = RobertUncaseQa(
                robert_path=config.MODEL_PATH, 
                conf=config.MODEL_CONF, 
                embedding_size=config.EMBEDDING_SIZE, 
                cnn_output_channel=config.CNN_OUTPUT_CHANNEL,
                kernel_width=config.CNN_KERNEL_WIDTH, 
                dropout_rate=config.DROPOUT_RATE
            ).to(config.DEVICE)

    model5.load_state_dict(torch.load(f'../model/{file_path}/fold5.pth'))

    return model1, model2, model3, model4, model5

if __name__ == '__main__':
    IF_UPLOAD = True
    IF_DOWNLOAD = False
    FILE = '0614_1'
    
    #upload model
    if IF_UPLOAD:
        for file_name in os.listdir(f"../model/{FILE}"):
            local_file = os.path.join(f"../model/{FILE}", file_name)
            upload_to_aws(local_file, config.BUCKET_NAME, f"{FILE}/{file_name}")

    #download model 
    if IF_DOWNLOAD:
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                        aws_secret_access_key=SECRET_KEY)
        all_objects = s3.list_objects(Bucket=config.BUCKET_NAME) 
        for item in all_objects['Contents']:
            model_name = item['Key']
            print(model_name)
            download_from_aws(config.BUCKET_NAME, model_name, "../model/" + model_name)
