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
            
        model.eval()
        fin_output_start = []
        fin_output_end = []
        fin_orig_sentiment = []
        #fin_padding_len = []
        #fin_text_token = []
        fin_origin_text = []
        fin_orig_selected = []
        fin_offset = []

        for bi, d in enumerate(data_loader):
            ids = d['ids']
            mask_id = d['mask_id']
            token_type_id = d['token_type_id']
            targets_start = d['targets_start']
            targets_end = d['targets_end']
            origin_sentiment = d['origin_sentiment']
            #padding_len = d['padding_len']
            #text_token = d['text_token']
            selected_text = d['origin_selected_text']
            origin_text = d['origin_text']
            offsets = d['offset'].numpy()
            fin_offset.append(offsets)

            with torch.no_grad():
                ids = ids.to(device, dtype=torch.long)
                mask_id = mask_id.to(device, dtype=torch.long)
                token_type_id = token_type_id.to(device, dtype=torch.long)
                targets_start = targets_start.to(device, dtype=torch.float)
                targets_end = targets_end.to(device, dtype=torch.float)

                output_start, output_end = model(ids, mask_id, token_type_id)

                fin_output_start.append(torch.softmax(output_start, axis=1).cpu().detach().numpy())
                fin_output_end.append(torch.softmax(output_end, axis=1).cpu().detach().numpy())
                #fin_padding_len.extend(padding_len.cpu().detach().numpy().tolist())
                #fin_text_token.extend(text_token)
                fin_orig_sentiment.extend(origin_sentiment)
                fin_orig_selected.extend(selected_text)
                fin_origin_text.extend(origin_text)

        fin_offset = np.vstack(fin_offset)
        fin_output_start = np.vstack(fin_output_start)
        fin_output_end = np.vstack(fin_output_end)

        output = []
        for j in range(len(fin_output_start)):
            #text_token = fin_text_token[j]
            #padding_len = fin_padding_len[j]
            origin_selected = fin_orig_selected[j]
            origin_sentiment = fin_orig_sentiment[j]
            origin_text = fin_origin_text[j]
            offset = fin_offset[j]
            start_idx = fin_output_start[j]
            end_idx = fin_output_end[j]
            start_idx = np.argmax(start_idx)
            end_idx = np.argmax(end_idx)
            # start_idx = np.nonzero(start_idx)[0][0]
            # end_idx = np.nonzero(end_idx)[0][0]

            if end_idx < start_idx:
                end_idx = start_idx
            
            final_output = ""
            for ix in range(start_idx, end_idx + 1):
                final_output += origin_text[offset[ix][0]:offset[ix][1]]
                if (ix+1) < len(offset) and offset[ix][1] < offset[ix+1][0]:
                    final_output += " "

            if origin_sentiment == 'neutral' or len(origin_text.split()) < 4:
                final_output = origin_text

            output.append(final_output)

        return output




if __name__ == '__main__':
    model = BertUncasedQa(BERT_PATH).to(DEVICE)
    MODEL_PATH = '../model/model_fold1.pth'
    device = 'cuda'
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    test_df = pd.read_csv("../input/test.csv")
    sample_submission = pd.read_csv("../input/sample_submission.csv")
    test_df.loc[:, "selected_text"] = test_df.text.values
    preds = predict(test_df)
    print(preds)
    sample_submission.loc[:, "selected_text"] = preds
    sample_submission.to_csv("../output/submission.csv", index=False)
    print(sample_submission.head())
