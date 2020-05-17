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

def load_model():
    model1 = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
    model1.load_state_dict(torch.load('../model/model_fold1.pth'))

    model2 = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
    model2.load_state_dict(torch.load('../model/model_fold2.pth'))

    model3 = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
    model3.load_state_dict(torch.load('../model/model_fold3.pth'))

    model4 = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
    model4.load_state_dict(torch.load('../model/model_fold4.pth'))

    model5 = RobertUncaseQa(config.MODEL_PATH).to(config.DEVICE)
    model5.load_state_dict(torch.load('../model/model_fold5.pth'))

    return model1, model2, model3, model4, model5

def predict(df):
    test_dataset = TweetDataset(
        text=df['text'].values,
        selected_text=df['selected_text'].values,
        sentiment=df['sentiment'].values,
        tokenizer=config.TOKENIZER,
        max_len=config.MAX_LEN,
        model_type=config.MODEL_TYPE
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, 
        batch_size = config.BATCH_SIZE
    ) 

    preds = pred_loop_fn(test_data_loader, config.DEVICE)


if __name__ == '__main__':

    test_df = pd.read_csv("../input/test.csv")
    test_df.loc[:, "selected_text"] = test_df.loc[:, 'text']
    sample_submission = pd.read_csv("../input/sample_submission.csv")
    preds = predict(test_df)

    test_df.loc[:, "selected_text"] = preds
    print(test_df.head())
