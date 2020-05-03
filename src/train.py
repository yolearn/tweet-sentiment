from engine import trn_loop_fn, eval_loop_fn
import pandas as pd 
from config import TRAIN_FILE, TOKENIZER, MAX_LEN, BATCH_SIZE
import torch
from dataload import TweetDataset
from sklearn import model_selection
from model import BertUncasedQa
from transformers import AdamW
from config import *

def run():
    df = pd.read_csv(TRAIN_FILE)
    #df = df[df['sentiment']!= 'neutral']
    # df['sentiment'] = df['sentiment'].apply(
    #                     lambda x : 1 if x=='positive' else 0
    # )

    trn_df, val_df = model_selection.train_test_split(df, 
                            random_state=42, 
                            test_size=0.2, 
                            stratify=df.sentiment.values
    )

    model = BertUncasedQa(BERT_PATH).to(DEVICE)
    tokenizer = TOKENIZER
    optimzer = AdamW(model.parameters(), lr=LR)
    
    trn_dataset = TweetDataset(
        text=trn_df['text'].values,
        selected_text=trn_df['selected_text'].values,
        sentiment=trn_df['sentiment'].values,
        tokenizer=TOKENIZER,
        max_len=MAX_LEN
    )

    val_dataset = TweetDataset(
        text=val_df['text'].values,
        selected_text=val_df['selected_text'].values,
        sentiment=val_df['sentiment'].values,
        tokenizer=TOKENIZER,
        max_len=MAX_LEN
    )

    trn_data_loader = torch.utils.data.DataLoader(
        dataset = trn_dataset, 
        batch_size = BATCH_SIZE
    )   

    val_data_loader = torch.utils.data.DataLoader(
        dataset = val_dataset, 
        batch_size = BATCH_SIZE
    ) 

    best_score = 0
    for i in range(EPOCH):
        trn_loop_fn(trn_data_loader, model, optimzer, DEVICE)
        cur_score = eval_loop_fn(val_data_loader, model, DEVICE)
        print(f"{i+1} EPOCH : JACCARDS = {cur_score}")
        if cur_score > best_score:
            torch.save(model.state_dict(), MODEL_PATH)
            best_score = cur_score
            
if __name__ == '__main__':
    run()
    