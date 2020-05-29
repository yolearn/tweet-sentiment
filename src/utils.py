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
            if config.SAVE_MODEL:
                self.save_model()

        else :
            self.counter+=1
            print(f'EarlyStopping counter : {self.counter} out of {self.patience}')
            
        if self.counter > self.patience:
            self.earlystop = True
            print(f'The best epoch is {self.epoch}')
            print(f'The best score is {self.max}')

    def save_model(self):
        #using parrallel will prefix "module."
        print('saving.....')
        try :
            torch.save(self.model.module.state_dict(), self.path)
        except:
            torch.save(self.model.state_dict(), self.path)
    

def cal_jaccard(fin_output_start, fin_output_end, fin_offset, fin_orig_sentiment, fin_orig_selected, fin_orig_text):
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

        if len(orig_text.split()) < 4:
            output_string = orig_text
        
        output_string = post_process(output_string)
        output_string = output_string.strip()
        if orig_sentiment == 'neutral':
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
        'pos_score' : sum(pos_score) / len(pos_score) if len(neu_score)!=0 else None,
        'neg_score' : sum(neg_score) / len(neg_score if len(neu_score)!=0 else None)
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
    print(fin_output_sentiment)
    print(fin_targ_sentiment)
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


def pre_process(df):
    #print(df.head())
    # df['text'] = df['text'].apply(lambda x:clean_text(x))
    # df['selected_text'] = df['selected_text'].apply(lambda x:clean_text(x))
    df = df[df['sentiment'] != 'neutral']

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



def load_model():
    model1 = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
    model1.load_state_dict(torch.load('../model/model_fold1.pth'))
    model1 = nn.DataParallel(model1)
    model1.eval()

    model2 = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
    model2.load_state_dict(torch.load('../model/model_fold2.pth'))
    model2 = nn.DataParallel(model2)
    model2.eval()

    model3 = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
    model3.load_state_dict(torch.load('../model/model_fold3.pth'))
    model3 = nn.DataParallel(model3)
    model3.eval()

    model4 = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
    model4.load_state_dict(torch.load('../model/model_fold4.pth'))
    model4 = nn.DataParallel(model4)
    model4.eval()

    model5 = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
    model5.load_state_dict(torch.load('../model/model_fold5.pth'))
    model5 = nn.DataParallel(model5)
    model5.eval()

    return model1, model2, model3, model4, model5





if __name__ == '__main__':
    IF_UPLOAD = True
    IF_DOWNLOAD = False
    #upload model
    if IF_UPLOAD:
        for file_name in os.listdir("../model"):
            local_file = os.path.join("../model", file_name)
            upload_to_aws(local_file, config.BUCKET_NAME, file_name)

    #download model 
    if IF_DOWNLOAD:
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                        aws_secret_access_key=SECRET_KEY)
        all_objects = s3.list_objects(Bucket=config.BUCKET_NAME) 
        for item in all_objects['Contents']:
            model_name = item['Key']
            print(model_name)
            download_from_aws(config.BUCKET_NAME, model_name, "../model/" + model_name)
