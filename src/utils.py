import time
import boto3
from botocore.exceptions import NoCredentialsError
import os
import config
import torch

ACCESS_KEY = 'AKIA3QRKIV7O4EO7EINY'
SECRET_KEY = '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'

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

    def __call__(self, cur, model, epoch):
        self.cur = cur
        self.model = model

        if self.cur > self.max:
            self.max = self.cur
            self.counter = 0
            self.epoch = epoch
            self.save_model()

        else :
            self.counter+=1
            print(f'EarlyStopping counter : {self.counter} out of {self.patience}')
            
        if self.counter > self.patience:
            self.earlystop = True
            print(f'The best epoch is {self.epoch}')
            print(f'The best score is {self.max}')

    def save_model(self):
        torch.save(self.model.state_dict(), self.path)


if __name__ == '__main__':
    #upload model
    for file_name in os.listdir("../model"):
        local_file = os.path.join("../model", file_name)
        
        upload_to_aws(local_file, config.BUCKET_NAME, file_name)