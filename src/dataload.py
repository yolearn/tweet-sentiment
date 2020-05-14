import torch
import transformers
import numpy as np
from config import *
import pandas as pd
"""
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

        char_targets = [0] * len(text)
        for j in range(idx0, idx1+1):
            if text[j] == ' ':
                char_targets[j] = 0
            else:
                char_targets[j] = 1

        #tok_ids : token_id, ex:101, 1045
        #tok_token : token, ex:'[CLS]', 'i',
        #tok_offset : (0, 0), (1, 2)
        tok_outputs = self.tokenizer.encode(text)
        tok_ids = tok_outputs.ids
        text_token = tok_outputs.tokens
        tok_offset =  tok_outputs.offsets[1:-1]

        targets_idx = [] 
        for j, (offset1, offset2) in enumerate(tok_offset):
            if sum(char_targets[offset1:offset2]) > 0:
                targets_idx.append(j)

        
        targets_start = np.zeros(self.max_len)
        targets_start[targets_idx[0]+1] = 1
        targets_end = np.zeros(self.max_len)
        targets_end[targets_idx[-1]+1] = 1

        #tok_ids = tok_ids 
        token_type_id = [0] + [0] * (len(tok_ids)-2) + [0]
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
            "mask_ids" : torch.tensor(mask, dtype=torch.long),
            "token_type_ids" : torch.tensor(token_type_id, dtype=torch.long),
            "targets_start" : torch.tensor(targets_start, dtype=torch.long),
            "targets_end" : torch.tensor(targets_end, dtype=torch.long),
            "orig_sentiment" : self.sentiment[item],
            "orig_text" : self.text[item],
            "orig_selected_text" : self.selected_text[item],
            "offset" : torch.tensor(tok_offset, dtype=torch.long)
        }


class RoTweetDataset:
    def __init__(self, text, selected_text, sentiment, tokenizer, max_len):
        self.text = text
        self.selected_text = selected_text
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __str__(self):
        return str(self.__getitem__(0)['text'])

    def __getitem__(self, item):
        text = " ".join(str(self.text[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())
        len_sel_text = len(selected_text)

        idx0 = None
        idx1 = None
        for i in (i for i, e in enumerate(text) if e == selected_text[0]):
            if text[i:i+len_sel_text] == selected_text:
                idx0 = i
                idx1 = i+len_sel_text-1
                break

             
        char_targets = [0]*len(text)
        for i in range(idx0, idx1+1):
            if text[i] != ' ':
                char_targets[i] = 1


        tok_output = self.tokenizer.encode(text)
        
        orig_ids = tok_output.ids
        offsets = tok_output.offsets

        targets_idx = []

        for i, (start, end) in enumerate(offsets):
            if sum(char_targets[start:end]) > 0:
                targets_idx.append(i)

        targets_start = np.zeros(self.max_len)
        targets_end = np.zeros(self.max_len)
        targets_start[targets_idx[0]+1] = 1
        targets_end[targets_idx[-1]+1] = 1

        
        ids = [0] + orig_ids + [2] 
        token_type_ids = [0] + [0] * len(orig_ids) + [0]
        mask_ids = len(token_type_ids) * [1]
        offsets = [(0,0)] + offsets + [(0,0)]

        assert len(ids) == len(token_type_ids)
        assert len(token_type_ids) == len(mask_ids)
        assert len(mask_ids) == len(offsets)

        padding_len = self.max_len - len(ids)

        if padding_len > 0:
            ids += [0] * padding_len
            token_type_ids += [0] * padding_len
            mask_ids += [0] * padding_len
            offsets += [(0,0)] * padding_len

        return {
            'ids':torch.tensor(ids, dtype=torch.long), 
            'token_type_ids' : torch.tensor(token_type_ids, dtype=torch.long), 
            'mask_ids' : torch.tensor(mask_ids, dtype=torch.long), 
            'offset' : torch.tensor(offsets, dtype=torch.long), 
            'orig_sentiment' : self.sentiment[item],
            'orig_selected_text' : self.selected_text[item],
            'orig_text' : self.text[item],
            'targets_start' : torch.tensor(targets_start, dtype=torch.long),
            'targets_end' : torch.tensor(targets_end, dtype=torch.long),
        }
"""


class TweetDataset:
    def __init__(self, text, selected_text, sentiment, tokenizer, max_len, model_type):
        self.text = text
        self.selected_text = selected_text
        self.sentiment = sentiment
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        # should return start and end 
        text = " ".join(str(self.text[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())
        sentiment = self.sentiment[item]


        sentiment_d = {
            'positive': 1313,
            'negative': 2430,
            'neutral': 7974
        }

        len_sel_text = len(selected_text)
        char_start_idx = None 
        char_end_idx = None

        for i in range(len(text)):
            if text[i:i+len_sel_text] == selected_text and text[i] == selected_text[0]:
                char_start_idx = i 
                char_end_idx = i+len_sel_text

        char_targets = np.zeros(len(text))
        for i in range(char_start_idx, char_end_idx):
            if text[i] != ' ':
                char_targets[i] = 1


        if self.model_type == 'roberta':
            token_out = self.tokenizer.encode(text)
            ids = token_out.ids 
            token_type_ids = token_out.type_ids 
            offsets = token_out.offsets 
            mask_ids = token_out.attention_mask

        elif self.model_type == 'bert':
            token_out = self.tokenizer.encode(text)
            ids = token_out.ids 
            token_type_ids = token_out.type_ids 
            offsets = token_out.offsets 
            mask_ids = token_out.attention_mask

        assert len(ids) == len(token_type_ids) == len(offsets) == len(mask_ids)
    
        targets_idx = []
        for i, (start, end) in enumerate(offsets):  
            if sum(char_targets[start:end]) > 0:
                targets_idx.append(i)

        target_start_idx = np.zeros(self.max_len)
        target_end_idx = np.zeros(self.max_len)
        # Due to [CLS]+1
        target_start_idx[targets_idx[0]+4] = 1
        target_end_idx[targets_idx[-1]+4] = 1
        
        """
        Robert
        - single input : <s> X </s>
        - pair input : <s> X </s></s> Y </s>

        Bert
        - single input : [CLS] X [SEP]
        - pair input : [CLS] X [SEP] Y [SEP]
        """

        if self.model_type == 'roberta':
            ids = [0] + [sentiment_d[sentiment]] + [2] +[2] + ids + [2]
            token_type_ids = [0] * 4 + token_type_ids + [0]
            mask_ids = [1] * 4 + mask_ids + [1]
            offsets = [(0,0)] * 4 + offsets + [(0,0)]
        
        elif self.model_type == 'bert':
            pass

        assert len(ids) == len(token_type_ids) == len(offsets) == len(mask_ids)

        if len(ids) < self.max_len:
            padding_len = self.max_len - len(ids)
            ids =  ids + [0] * padding_len
            token_type_ids = token_type_ids + [0]*padding_len
            mask_ids = mask_ids + [0] * padding_len
            offsets = offsets + [(0,0)] * padding_len

        return {
            'ids' : torch.tensor(ids, dtype=torch.long),
            'mask_ids' : torch.tensor(mask_ids, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_type_ids, dtype=torch.long),
            'target_start_idx' : torch.tensor(target_start_idx, dtype=torch.long),
            'target_end_idx' : torch.tensor(target_end_idx, dtype=torch.long),
            'offsets' : torch.tensor(offsets, dtype=torch.long),
            'orig_sentiment' : sentiment,
            'orig_sele_text' : selected_text,
            'orig_text' : text,
        }



if __name__ == "__main__":
    trn_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv("../input/test.csv")
    test_df['selected_text'] = 'temp'

    # dataset = TweetDataset(trn_df['text'].values, 
    #                     trn_df['selected_text'].values, 
    #                     trn_df['sentiment'].values, 
    #                     TOKENIZER, 
    #                     MAX_LEN
    #                 )

    # dataset = TweetDataset(test_df['text'].values, 
    #                 test_df['selected_text'].values, 
    #                 test_df['sentiment'].values, 
    #                 TOKENIZER, 
    #                 MAX_LEN
    #             )
    # print(dataset[0]['offset'].shape)
    # print(dataset[1]['offset'].shape)
    # print(dataset[2]['offset'].shape)
    # for i in range(1000):
    #     print(dataset[i]['origin_sentiment'])

    trn_df = trn_df[trn_df['sentiment']=='positive']
    # dataset = RoTweetDataset(trn_df['text'].values, 
    #                     trn_df['selected_text'].values, 
    #                     trn_df['sentiment'].values, 
    #                     ROBERT_TOKENIZER, 
    #                     MAX_LEN
    #                 )
    dataset = TweetDataset(trn_df['text'].values, 
                    trn_df['selected_text'].values, 
                    trn_df['sentiment'].values, 
                    TOKENIZER, 
                    MAX_LEN,
                    'roberta'
                )            
    
    print(dataset[0])


