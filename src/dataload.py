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
        return len(self.sentiment)

    def __getitem__(self, item):
        text = " ".join(str(self.text[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())
        len_sel_text = len(selected_text)

        idx0 = -1
        idx1 = -1
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

        #tok_ids : token_id, ex:101, 1045
        #tok_token : token, ex:'[CLS]', 'i',
        #tok_offset : (0, 0), (1, 2)
        tok_outputs = self.tokenizer.encode(text)
        tok_ids = tok_outputs.ids
        text_token = tok_outputs.tokens
        tok_offset =  tok_outputs.offsets[1:-1]

        targets_idx = [] 
        for j, (offset1, offset2) in enumerate(tok_offset):
            if sum(char_target[offset1:offset2]) > 0:
                targets_idx.append(j)

        
        targets_start = np.zeros(MAX_LEN)
        targets_start[targets_idx[0]+1] = 1
        targets_end = np.zeros(MAX_LEN)
        targets_end[targets_idx[-1]+1] = 1

        tok_ids = tok_ids 
        token_type_id = [0] + [1] * (len(tok_ids)-2) + [1]
        mask = [1] * len(tok_ids)
        tok_offset = [(0,0)] + tok_offset + [(0,0)]

        assert len(mask) == len(token_type_id)
        assert len(tok_offset) == len(token_type_id) 

        padding_len = self.max_len - len(tok_ids)
        mask = mask + [0] * padding_len 
        tok_ids = tok_ids + [0] * padding_len
        token_type_id = token_type_id + [0] * padding_len 
        tok_offset = tok_offset + ([(0, 0)] * padding_len)
        # targets = targets + [0] * padding_len 
        # targets_start = targets_start + [0] * padding_len 
        # targets_end = targets_end + [0] * padding_len 

        
        return {
            "ids" : torch.tensor(tok_ids, dtype=torch.long),
            "mask_id" : torch.tensor(mask, dtype=torch.long),
            "token_type_id" : torch.tensor(token_type_id, dtype=torch.long),
            #"targets" : torch.tensor(targets, dtype=torch.long),
            "targets_start" : torch.tensor(targets_start, dtype=torch.long),
            "targets_end" : torch.tensor(targets_end, dtype=torch.long),
            #"padding_len" : padding_len,
            #"text_token" : " ".join(text_token),
            #"sentiment" : torch.tensor(sentiment, dtype=torch.float),
            "origin_sentiment" : self.sentiment[item],
            "origin_text" : self.text[item],
            "origin_selected_text" : self.selected_text[item],
            "offset" : torch.tensor(tok_offset, dtype=torch.long)
        }
        
if __name__ == "__main__":
    trn_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv("../input/test.csv")
    test_df['selected_text'] = 'temp'

    dataset = TweetDataset(trn_df['text'].values, 
                        trn_df['selected_text'].values, 
                        trn_df['sentiment'].values, 
                        TOKENIZER, 
                        MAX_LEN
                    )

    dataset = TweetDataset(test_df['text'].values, 
                    test_df['selected_text'].values, 
                    test_df['sentiment'].values, 
                    TOKENIZER, 
                    MAX_LEN
                )
    print(dataset[0]['offset'].shape)
    print(dataset[1]['offset'].shape)
    print(dataset[2]['offset'].shape)
    # for i in range(1000):
    #     print(dataset[i]['origin_sentiment'])
