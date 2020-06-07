import torch
import transformers
import numpy as np
import pandas as pd
from utils import SentencePieceTokenizer, clean_text
import sentencepiece_pb2


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


        if self.model_type in ['roberta-base', 'roberta-base-squad2', 'roberta-large']:
            token_out = self.tokenizer.encode(text)
            ids = token_out.ids 
            token_type_ids = token_out.type_ids 
            offsets = token_out.offsets 
            mask_ids = token_out.attention_mask

            sentiment_d = {
                'positive': 1313,
                'negative': 2430,
                'neutral': 7974
            }
        
        elif self.model_type in ['bert-base-uncased']:
            token_out = self.tokenizer.encode(text)
            ids = token_out.ids
            token_type_ids = token_out.type_ids 
            offsets = token_out.offsets
            mask_ids = token_out.attention_mask

            sentiment_d = {
                'positive': 3893,
                'negative': 4997,
                'neutral': 8699
            }

        elif self.model_type in ['albert-base-v2']:
            token_out = self.tokenizer.encode(text)
            ids = token_out[0]
            offsets = token_out[1]
            token_type_ids = len(ids) * [1]
            mask_ids = len(ids) * [1]
            
            sentiment_d = {
                'positive': 2221,
                'negative': 3682,
                'neutral': 8387
            }


        assert len(ids) == len(token_type_ids) == len(offsets) == len(mask_ids)
    
        targets_idx = []
        for i, (start, end) in enumerate(offsets):  
            if sum(char_targets[start:end]) > 0:
                targets_idx.append(i)

        target_start_idx = np.zeros(self.max_len)
        target_end_idx = np.zeros(self.max_len)
        # target_start_idx = 0
        # target_end_idx = 0

        """
        Robert
        - single input : <s> X </s>
        - pair input : <s> X </s></s> Y </s>

        Bert
        - single input : [CLS] X [SEP]
        - pair input : [CLS] X [SEP] Y [SEP]
        
        Albert
        - single input : [CLS] X [SEP]
        - pair input : [CLS] X [SEP] Y [SEP]
        """
        

        if self.model_type in ['roberta-base', 'roberta-base-squad2', 'roberta-large']:
            # target_start_idx = targets_idx[0]+4
            # target_start_idx = targets_idx[-1]+4
            target_start_idx[targets_idx[0]+4] = 1
            target_end_idx[targets_idx[-1]+4] = 1

            ids = [0] + [sentiment_d[sentiment]] + [2] +[2] + ids + [2]
            token_type_ids = [0] * 4 + token_type_ids + [0]
            mask_ids = [1] * 4 + mask_ids + [1]
            offsets = [(0,0)] * 4 + offsets + [(0,0)]
        
        elif self.model_type in ['bert-base-uncased']:
            # target_start_idx = targets_idx[0]+3
            # target_start_idx = targets_idx[-1]+3
            target_start_idx[targets_idx[0]+3] = 1
            target_end_idx[targets_idx[-1]+3] = 1

            ids = [101] + [sentiment_d[sentiment]] + [102] + ids + [102]
            token_type_ids = [0] * 3 + [1] * len(token_type_ids) + [1]
            mask_ids = [1] * 3 + mask_ids + [1]
            offsets = [(0,0)] * 3 + offsets + [(0,0)]

        elif self.model_type in ['albert-base-v2']:
            # target_start_idx = targets_idx[0]+3
            # target_start_idx = targets_idx[-1]+3
            target_start_idx[targets_idx[0]+3] = 1
            target_end_idx[targets_idx[-1]+3] = 1

            ids = [2] + [sentiment_d[sentiment]] + [3] + ids + [3]
            token_type_ids = [0] * 3 + [1] * len(token_type_ids) + [1]
            mask_ids = [1] * 3 + mask_ids + [1]
            offsets = [(0,0)] * 3 + offsets + [(0,0)]

        assert len(ids) == len(token_type_ids) == len(offsets) == len(mask_ids)

        if len(ids) < self.max_len:
            padding_len = self.max_len - len(ids)
            ids =  ids + [0] * padding_len
            token_type_ids = token_type_ids + [0]*padding_len
            mask_ids = mask_ids + [0] * padding_len
            offsets = offsets + [(0,0)] * padding_len

        if sentiment == 'positive':
            targ_sentiment = 0
        elif sentiment == 'negative':
            targ_sentiment = 1
        elif sentiment == 'neutral':
            targ_sentiment = 2

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
            'targ_sentiment' : torch.tensor(targ_sentiment, dtype=torch.long)
        }


if __name__ == "__main__":
    trn_df = pd.read_csv(config.TRAIN_FILE)
    trn_df['text'] = trn_df['text'].apply(lambda x:clean_text(x))
    trn_df['selected_text'] = trn_df['selected_text'].apply(lambda x:clean_text(x))
    # test_df = pd.read_csv("../input/test.csv")
    # test_df['selected_text'] = 'temp'

    # trn_df = trn_df[trn_df['sentiment']=='positive']

    dataset = TweetDataset(trn_df['text'].values, 
                    trn_df['selected_text'].values, 
                    trn_df['sentiment'].values, 
                    config.TOKENIZER, 
                    config.MAX_LEN,
                    config.MODEL_TYPE
                )            
    
    for i in range(len(trn_df)):
        try:
            dataset[i]
        except:
            print(i)

    
 

