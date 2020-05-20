import torch
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import config

class CrossValidation:
    def __init__(self, df, split_type, seed, nfolds, shuffle=True, trn_rate=0.8):
        self.df = df
        self.split_type = split_type
        self.seed = seed
        self.nfolds = nfolds
        self.shuffle = shuffle
        self.trn_rate = trn_rate

    def split(self):
        if self.split_type == 'kfold':
            kf = KFold(n_splits=self.nfolds, shuffle=self.shuffle)
            for i, (trn_idx, val_idx) in enumerate(kf.split(self.df)):
                print(f"{i+1} fold : ")
                yield trn_idx, val_idx

        elif self.split_type == 'skfold':
            skf = StratifiedKFold(n_splits=self.nfolds, shuffle=self.shuffle)
            for i, (trn_idx, val_idx) in enumerate(skf.split(self.df)):
                print(f"{i+1} fold : ")
                yield trn_idx, val_idx
        
        elif self.split_type == 'pure_split':
            trn_df, val_df = train_test_split(self.df, train_size=self.trn_rate, random_state=config.SEED)
            yield trn_df.index, val_df.index


if __name__ == "__main__":
    #orig_df = pd.read_csv(config.TRAIN_FILE).dropna()
    df = pd.read_csv(config.TRAIN_FILE).copy(deep=True)
    #cv = CrossValidation(df, config.SPLIT_TYPE, config.SEED, config.NFOLDS, config.SHUFFLE)
    cv = CrossValidation(df, 'pure_split', config.SEED, config.NFOLDS, config.SHUFFLE)
    
    for fold, (trn_idx, val_indx) in enumerate(cv.split()):
        print(fold)
        print(len(df.iloc[trn_idx]))
        print(len(df.iloc[val_indx]))