import pandas as pd
import os
import tokenizers
import numpy as np
import torch
from scipy.special import softmax
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils import jaccard, post_process
from cross_val import CrossValidation

def encode(text, tokenizer, sentiment, output_start, output_end):
    text = " ".join(str(text).split())

    token_out = tokenizer.encode(text)
    offsets = token_out.offsets 

    offsets = [(0,0)] * 4 + offsets + [(0,0)]
    if len(offsets) < MAX_LEN:
        padding_len = MAX_LEN - len(offsets)
        offsets = offsets + [(0,0)] * padding_len

    if output_start > output_end:
        output_end = output_start
    try:
        output_string = ''
        for j in range(output_start, output_end+1):
            output_string += text[offsets[j][0]:offsets[j][1]] 
            if (j+1) < len(offsets) and offsets[j][1] < offsets[j+1][0]:
                    output_string += " "
    except:
       pass

    if sentiment == 'neutral' or len(text.split()) < 4:
        output_string = text

    #output_string = post_process(output_string)
    output_string = output_string.strip()

    return output_string 



def filt_thresh(idx, thresh=0):
    idx = idx[idx > thresh]
    
    return np.nonzero(idx)[0] if len(np.nonzero(idx)[0]) > 0 else np.array([0])

BERT_TOKENIZER = tokenizers.BertWordPieceTokenizer(
    "../input/vocab.txt",
    lowercase=True,
    add_special_tokens = False
)
ROBERT_TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    vocab_file="../input/roberta-base/vocab.json",
    merges_file="../input/roberta-base/merges.txt",
    lowercase=True,
    add_prefix_space=True
)




CV_FILE = '../model/exper/cv.csv'
MAX_LEN = 100

df = pd.read_csv("../input/train.csv")
print(df.keys())
pred_df = pd.read_csv(CV_FILE).dropna()
print(pred_df.keys())

df['text'] = df['text'].astype(str)
assert len(df) == len(pred_df)
df = pd.concat([df, pred_df], axis=1)

trn_df, val_df = train_test_split(df, train_size=0.8, random_state=42)
score = []
for df in [trn_df, val_df]:
    for row in df.itertuples():
        text = row[2]
        selected_text = row[3]
        sentiment = row[4]
        start_idx = np.array(row[5:105])
        end_idx = np.array(row[105:205])

        start_idx = np.argmax(start_idx)
        end_idx = np.argmax(end_idx)
        
        output_string = encode(text, ROBERT_TOKENIZER, sentiment, start_idx, end_idx)
        score.append(jaccard(selected_text, output_string))

    print("softmax : ", sum(score) / len(score))


threshs = [0.00001+0.00000003*i for i in range(1,10000) if 0.00001+0.00000003*i < 0.8]

val_curr_score = 0
val_max_score = 0
bst_threh = 0



for thresh in tqdm(threshs):
    trn_score = []
    val_score = []
    for row in trn_df.itertuples():

        text = row[2]
        selected_text = row[3]
        sentiment = row[4]
        start_idx = np.array(row[5:105])
        end_idx = np.array(row[105:205])


        start_idx = filt_thresh(start_idx, thresh)
        end_idx = filt_thresh(end_idx, thresh)
        start_idx = start_idx[0]
        end_idx = end_idx[-1]

        # start_idx = np.argmax(start_idx)
        # end_idx = np.argmax(end_idx)
        
        output_string = encode(text, ROBERT_TOKENIZER, sentiment, start_idx, end_idx)
        trn_score.append(jaccard(selected_text, output_string))
    
    trn_curr_score = round(sum(trn_score) / len(trn_score), 3)
    
    for row in val_df.itertuples():
        text = row[2]
        selected_text = row[3]
        sentiment = row[4]
        start_idx = np.array(row[5:105])
        end_idx = np.array(row[105:205])

        start_idx = filt_thresh(start_idx, thresh)
        end_idx = filt_thresh(end_idx, thresh)
        start_idx = start_idx[0]
        end_idx = end_idx[-1]

        # start_idx = np.argmax(start_idx)
        # end_idx = np.argmax(end_idx)
        
        output_string = encode(text, ROBERT_TOKENIZER, sentiment, start_idx, end_idx)
        val_score.append(jaccard(selected_text, output_string))


    val_curr_score = round(sum(val_score) / len(val_score), 3)
    #print(thresh, val_curr_score)
    if val_curr_score > val_max_score:
        bst_threh = thresh
        val_max_score = val_curr_score
        print('Best Thresh :', bst_threh)
        print('Train ', trn_curr_score)
        print("val", val_curr_score)
    
