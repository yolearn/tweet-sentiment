from engine import trn_loop_fn, eval_loop_fn
import pandas as pd 
import torch
from dataload import TweetDataset
from sklearn import model_selection
from model import BertUncasedQa, RobertUncaseQa
from transformers import AdamW
import config
from cross_val import cv

def run():
    #df = df[df['sentiment']!= 'neutral']
    # trn_df, val_df = model_selection.train_test_split(df, 
    #                     random_state=42, 
    #                     test_size=0.2, 
    #                     stratify=df.sentiment.values
    #     )
    # df['sentiment'] = df['sentiment'].apply(
    #                     lambda x : 1 if x=='positive' else 0
    # )
    for fold, (trn_df, val_df) in enumerate(cv.split()):
        #bert_model = BertUncasedQa(config.BERT_PATH).to(config.DEVICE)
        #tokenizer = BERT_TOKENIZER
        robert_model = RobertUncaseQa(config.ROBERT_PATH).to(config.DEVICE)
        model = robert_model
        tokenizer = config.ROBERT_TOKENIZER
        optimzer = AdamW(model.parameters(), lr=config.LR)

        trn_dataset = TweetDataset(
            text=trn_df['text'].values,
            selected_text=trn_df['selected_text'].values,
            sentiment=trn_df['sentiment'].values,
            tokenizer=tokenizer,
            max_len=config.MAX_LEN
        )

        val_dataset = TweetDataset(
            text=val_df['text'].values,
            selected_text=val_df['selected_text'].values,
            sentiment=val_df['sentiment'].values,
            tokenizer=tokenizer,
            max_len=config.MAX_LEN
        )

        trn_data_loader = torch.utils.data.DataLoader(
            dataset = trn_dataset, 
            batch_size = config.BATCH_SIZE
        )   

        val_data_loader = torch.utils.data.DataLoader(
            dataset = val_dataset, 
            batch_size = config.BATCH_SIZE
        ) 

        best_score = 0
        for i in range(config.EPOCH):
            trn_loop_fn(trn_data_loader, model, optimzer, config.DEVICE)
            cur_score = eval_loop_fn(val_data_loader, model, config.DEVICE)
            print(f"{i+1} EPOCH : JACCARDS = {cur_score}")
            if cur_score > best_score:
                torch.save(model.state_dict(), f'../model/model_fold{fold+1}_del.pth')
                best_score = cur_score
            
if __name__ == '__main__':
    #CUDA_VISIBLE_DEVICES=1 python3 train.py
    run()
       
    