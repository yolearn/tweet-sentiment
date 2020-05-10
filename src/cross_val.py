import torch
import pandas as pd
import numpy as np
import os
from config import *
from sklearn.model_selection import KFold

class CrossValidation:
    def __init__(self, df, split_type, seed, nfolds, shuffle=True):
        self.df = df
        self.split_type = split_type
        self.seed = seed
        self.nfolds = nfolds
        self.shuffle = shuffle
        

    def split(self):
        if self.split_type == 'kfold':
            kf = KFold(n_splits=self.nfolds, shuffle=self.shuffle)
            for i, (trn_ind, val_ind) in enumerate(kf.split(self.df)):
                print(f"{i+1} fold : ")
                yield (self.df.iloc[trn_ind], self.df.iloc[val_ind])

        elif self.split_type == 'ks':
            pass


df = pd.read_csv(TRAIN_FILE, nrows=1000)
# print(df.shape[0])
# df = df[df['sentiment'] != 'neutral']
# print(df.shape[0])
# print(df.head())
df['text'] = df['text'].astype(str)
df['selected_text'] = df['selected_text'].astype(str)
cv = CrossValidation(df, SPLIT_TYPE, SEED, NFOLDS, SHUFFLE)
cv_df = cv.split()

if __name__ == "__main__":
    SEED = 42
    NFOLDS = 5
    SHUFFLE = True
    SPLIT_TYPE = 'kfold'
    df = pd.read_csv(TRAIN_FILE)
    cv = CrossValidation(df, SPLIT_TYPE, SEED, NFOLDS, SHUFFLE)
    
    for fold, (trn_df, val_df) in enumerate(cv.split()):
        print(fold)
        print(trn_df.head())