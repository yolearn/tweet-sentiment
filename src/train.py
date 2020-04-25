from engine import train_on_batch
import pandas as pd 
from config import TRAIN_FILE, TOKENIZER, MAX_LEN, BATCH_SIZE
import torch
from dataload import TweetDataset
from sklearn import model_selection

def run():

    df = pd.read_csv(TRAIN_FILE, nrows=10)
    df['sentiment'] = df['sentiment'].apply(
        lambda x : 1 if x=='positive' else 0
    )
    trn_df, val_df = model_selection.train_test_split(df, 
                            random_state=42, 
                            test_size=0.2, 
                            stratify=df.sentiment.values
    )
    
    trn_dataset = TweetDataset(
        text=trn_df['text'],
        selected_text=trn_df['selected_text'],
        sentiment=trn_df['sentiment'],
        tokenizer=TOKENIZER,
        max_len=MAX_LEN
    )

    val_dataset = TweetDataset(
        text=val_df['text'],
        selected_text=val_df['selected_text'],
        sentiment=val_df['sentiment'],
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

    # for i in range(EPOCH):
    #     print(f"{i+1} epoch")
    #     train_on_batch(data_loader, model, optimzer, device))


if __name__ == '__main__':
    run()
    