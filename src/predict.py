import torch
from model import BertUncasedQa
from config import *
from dataload import TweetDataset
import pandas as pd
import string
from tqdm import tqdm
import numpy as np

def predict(df):
    with torch.no_grad():
        dataset = TweetDataset(
            text=df['text'].values,
            selected_text=df['selected_text'].values,
            sentiment=df['sentiment'].values,
            tokenizer=TOKENIZER,
            max_len=MAX_LEN
        )

        data_loader = torch.utils.data.DataLoader(
            dataset = dataset, 
            batch_size = BATCH_SIZE
        )  
        
        fin_output_start = []
        fin_output_end = []
        fin_orig_sentiment = []
        fin_padding_len = []
        fin_text_token = []
        fin_origin_text = []
        fin_orig_selected = []

        for bi, d in enumerate(data_loader):
            ids = d['ids']
            mask_id = d['mask_id']
            token_type_id = d['token_type_id']
            targets_start = d['targets_start']
            targets_end = d['targets_end']
            origin_sentiment = d['origin_sentiment']
            padding_len = d['padding_len']
            text_token = d['text_token']
            selected_text = d['origin_selected_text']
            origin_text = d['origin_text']

            ids = ids.to(DEVICE, dtype=torch.long)
            mask_id = mask_id.to(DEVICE, dtype=torch.long)
            token_type_id  = token_type_id.to(DEVICE, dtype=torch.long)
            targets_start = targets_start.to(DEVICE, dtype=torch.float)
            targets_end = targets_end.to(DEVICE, dtype=torch.float)

            output_start, output_end = model(ids, mask_id, token_type_id)

            fin_output_start.append(torch.sigmoid(output_start).cpu().detach().numpy())
            fin_output_end.append(torch.sigmoid(output_end).cpu().detach().numpy())
            fin_padding_len.extend(padding_len.cpu().detach().numpy().tolist())
            fin_text_token.extend(text_token)
            fin_orig_sentiment.extend(origin_sentiment)
            fin_orig_selected.extend(selected_text)
            fin_origin_text.extend(origin_text)

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    output = []
    for j in tqdm(range(len(fin_output_start))):
        orig_sentiment = fin_orig_sentiment[j]
        padding_len    = fin_padding_len[j]
        text_token     = fin_text_token[j]
        origin_text    = fin_origin_text[j]
        origin_selected = fin_orig_selected[j]


        if padding_len > 0:
            mask_start = fin_output_start[j][:-padding_len] >= THRESHOLD
            mask_end   = fin_output_end[j][:-padding_len] >= THRESHOLD
        else:
            mask_start = fin_output_start[i][:-padding_len]
            mask_end   = fin_output_end[i][:-padding_len]

        mask = [0]* len(mask_start)
        idx_start_l = np.nonzero(mask_start)[0]
        idx_end_l   = np.nonzero(mask_end)[0]

        if len(idx_start_l)>0:
            idx_start = idx_start_l[0]
            if len(idx_end_l) > 0:
                idx_end = idx_end_l[0]
            else:
                idx_end = idx_start 
        else:
            idx_start = idx_end = 0

        for i in range(idx_start, idx_end+1):
            mask[i] = 1


        output_token = [token for i, token in enumerate(text_token.split()) if mask[i]==1 and token not in ("[CLS]", "[SEP]")]
        
        final_output = ''
        for token in output_token:
            if token.startswith("##"):
                final_output+=token[2:]
            elif len(token) == 1 or token in string.punctuation:
                final_output+=token
            else:
                final_output+=' '
                final_output+=token
            
        output.append(final_output.strip())

    return output




if __name__ == '__main__':
    model = BertUncasedQa(BERT_PATH).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    test_df = pd.read_csv("../input/test.csv")
    test_df.loc[:, "selected_text"] = test_df.text.values
    preds = predict(test_df)
    test_df.loc[:, "selected_text"] = preds
    test_df[['textID', 'selected_text']].to_csv("../output/submission.csv", index=True)
    print(test_df.head())
    # sub = pd.read_csv("../input/sample_submission.csv")
    # print(test_df.head())
    # print(sub.head())