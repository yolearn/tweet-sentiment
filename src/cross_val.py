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
        self.seed = seed

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
            kf = KFold(n_splits=self.nfolds, shuffle=self.shuffle)
            for i, (trn_idx, val_idx) in enumerate(kf.split(self.df)):
                print(f"{i+1} fold : ")
                yield trn_idx, val_idx
                break

if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_FILE).dropna()
    cv = CrossValidation(df, 'pure_split', config.SEED, config.NFOLDS, config.SHUFFLE)
    
    for fold, (trn_idx, val_indx) in enumerate(cv.split()):
        print(type(trn_idx))
        print(fold)
        print(len(df.iloc[trn_idx]))
        print(len(df.iloc[val_indx]))