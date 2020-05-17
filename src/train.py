import argparse
from engine import trn_loop_fn, eval_loop_fn
import pandas as pd 
import numpy as np
import torch
from dataload import TweetDataset
from sklearn import model_selection
from model import BertUncasedQa, RobertUncaseQa
from transformers import AdamW
import config
from cross_val import CrossValidation
import torch.nn as nn
from utils import upload_to_aws, EarlyStopping, set_seed
import time

def run(cv):
    score = []
    start_preds = np.zeros((df.shape[0], config.MAX_LEN))
    end_preds = np.zeros((df.shape[0], config.MAX_LEN))
    
    for fold, (trn_idx, val_idx) in enumerate(cv.split()):
        trn_df = df.iloc[trn_idx]
        val_df = df.iloc[val_idx]
        #bert_model = BertUncasedQa(config.BERT_PATH).to(config.DEVICE)
        #tokenizer = BERT_TOKENIZER
        robert_model = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
        model = robert_model
        model = nn.DataParallel(model)
        optimzer = AdamW(model.parameters(), lr=config.LR)

        trn_dataset = TweetDataset(
            text=trn_df['text'].values,
            selected_text=trn_df['selected_text'].values,
            sentiment=trn_df['sentiment'].values,
            tokenizer=config.TOKENIZER,
            max_len=config.MAX_LEN,
            model_type=config.MODEL_TYPE
        )

        val_dataset = TweetDataset(
            text=val_df['text'].values,
            selected_text=val_df['selected_text'].values,
            sentiment=val_df['sentiment'].values,
            tokenizer=config.TOKENIZER,
            max_len=config.MAX_LEN,
            model_type=config.MODEL_TYPE
        )

        trn_data_loader = torch.utils.data.DataLoader(
            dataset = trn_dataset, 
            batch_size = config.BATCH_SIZE
        )   

        val_data_loader = torch.utils.data.DataLoader(
            dataset = val_dataset, 
            batch_size = config.BATCH_SIZE
        ) 

        model_pth = f'../model/model_fold{fold+1}.pth'
        earlystop = EarlyStopping(path=model_pth, patience=config.PATIENCE)
   
        for i in range(config.EPOCH):       
            trn_loop_fn(trn_data_loader, model, optimzer, config.DEVICE)
            #cur_score = eval_loop_fn(trn_data_loader, model, config.DEVICE)
            #print(f"Train {i+1} EPOCH : JACCARDS = {cur_score}")
            cur_score, pred1, pred2 = eval_loop_fn(val_data_loader, model, config.DEVICE, 'val')

            earlystop(cur_score, model, i+1, pred1, pred2)
            if earlystop.earlystop:
                print("Early stopping")
                break

        start_preds[val_idx] = earlystop.pred1
        end_preds[val_idx] = earlystop.pred2
        score.append(earlystop.max)
                  
    print("cv score : ", score)
    print("average cv score : ", sum(score) / len(score))
    preds = np.concatenate([start_preds, end_preds], axis=1)
    pd.DataFrame(preds).to_csv('cv.csv', index=False)   

if __name__ == '__main__':
    #CUDA_VISIBLE_DEVICES=1 python3 train.py
    # parser = argparse.ArgumentParser(description="Let's tuning hyperparameter")
    # parser.add_argument('--batch_size', default=16)
    # parser.add_argument('--max_len', default=128)
    # parser.add_argument('--EPOCH', default=1)
    # parser.add_argument('--lr', default=3e-5)
    # parser.add_argument('--nfolds', default=5)
    # parser.add_argument('--split_type', default='kfold')
    # parser.add_argument('--patience', default=1)
    # parser.add_argument('--dropout_rate', default=0.3)

    # args = parser.parse_args()
    # args = dict(vars(args))

    set_seed()
    df = pd.read_csv(config.TRAIN_FILE).dropna()[:1000]
    df['text'] = df['text'].astype(str)
    df['selected_text'] = df['selected_text'].astype(str)
    cv = CrossValidation(df, config.SPLIT_TYPE, config.SEED, config.NFOLDS, config.SHUFFLE)
    run(cv)

    #model_save
    if False:
        for file_name in os.listdir("../model"):
            local_file = os.path.join("../model", file_name)
            upload_to_aws(local_file, config.BUCKET_NAME, file_name)
       
    