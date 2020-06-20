import torch
from model import BertUncasedQa, RobertUncaseQa
from config import *
from dataload import TweetDataset
import pandas as pd
import string
from tqdm import tqdm
import numpy as np
from engine import eval_loop_fn
import config
from scipy.special import softmax
from engine import pred_loop_fn


def predict(df):
    test_dataset = TweetDataset(
        text=df['text'].values,
        selected_text=df['selected_text'].values,
        sentiment=df['sentiment'].values,
        tokenizer=config.TOKENIZER,
        max_len=config.MAX_LEN,
        model_type=config.MODEL_VERSION
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, 
        batch_size = config.BATCH_SIZE
    ) 

    preds = pred_loop_fn(test_data_loader, config.DEVICE, FILE_PATH)
    return preds

if __name__ == '__main__':
    FILE_PATH = "0614_1"

    test_df = pd.read_csv("../input/test.csv")
    test_df.loc[:, "selected_text"] = test_df.loc[:, 'text']
    sample_submission = pd.read_csv("../input/sample_submission.csv")
    preds = predict(test_df)

    sample_submission.loc[:, "selected_text"] = preds
    sample_submission.to_csv('../output/submission/output.csv', index=False)
