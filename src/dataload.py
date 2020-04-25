import torch
import transformers
import numpy as np
from config import *
import pandas as pd

class TweetDataset():
    def __init__(self, text, selected_text, sentiment, tokenizer, max_len):
        self.text = text
        self.selected_text = selected_text
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return lem(self.sentiment)


    def __getitem__(self, item):
        text = str(self.text[item])
        selected_text = str(self.selected_text[item])
        

        len_sel_text = len(selected_text)


        for ind in (i for i, e in enumerate(text) if e == selected_text[0]):
            if text[ind:ind+len_sel_text] == selected_text:
                idx0 = ind    
                idx1 = ind+len_sel_text -1
                break
        

        char_target = [0] * len(text)
        for j in range(idx0, idx1+1):
            if text[j] == ' ':
                char_target[j] = 0
            else:
                char_target[j] = 1

        
        tok_outputs = self.tokenizer.encode(text)
        tok_ids = tok_outputs.ids
        tok_tokens = tok_outputs.tokens
        tok_offset =  tok_outputs.offsets[1:-1]

        targets = [0] * (len(tok_tokens)-2) 
        for j, (offset1, offset2) in enumerate(tok_offset):
            if sum(char_target[offset1:offset2]) > 1:
                targets[j] =1

        targets = [0] + targets + [0]  #cls sep
        targets_start = [0] * len(targets) 
        targets_end = [0] * len(targets)

        non_zero = np.nonzero(targets)[0]
        if len(np.nonzero(targets)) > 0:
            targets_start[non_zero[0]] = 1
            targets_start[non_zero[-1]] = 1

        mask = [1] * len(tok_ids)
        token_type_id = [0] * len(tok_ids)

        padding_len = self.max_len - len(tok_ids)
        mask = mask + [0] * padding_len 
        tok_ids = tok_ids + [0] * padding_len
        token_type_id = token_type_id + [0] * padding_len 
        targets = targets + [0] * padding_len 
        targets_start = targets_start + [0] * padding_len 
        targets_end = targets_end + [0] * padding_len 

        if self.sentiment[item] == 'positive':
            sentiment = [1,0,0]
        elif self.sentiment[item] == 'negative':
            sentiment = [0,0,1]
        else:
            sentiment = [0,1,0]

        print(tok_tokens)
        return {
            "token_id" : torch.tensor(tok_ids, dtype=torch.long),
            "mask_id" : torch.tensor(mask, dtype=torch.long),
            "token_type_id" : torch.tensor(token_type_id, dtype=torch.long),
            "targets" : torch.tensor(targets, dtype=torch.long),
            "targets_start" : torch.tensor(targets_start, dtype=torch.long),
            "targets_end" : torch.tensor(targets_end, dtype=torch.long),
            "padding_len" : padding_len,
            "sentiment" : torch.tensor(sentiment, dtype=torch.float),
            "origin_sentiment" : self.sentiment[item],
            "text_token" : " ".join(tok_tokens)
        }
        
if __name__ == "__main__":
    df = pd.read_csv(TRAIN_FILE)
    dataset = TweetDataset(df['text'], df['selected_text'], df['sentiment'], TOKENIZER, MAX_LEN)
    print(dataset[0])