import torch
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold

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
            for i, (trn_idx, val_idx) in enumerate(kf.split(self.df)):
                print(f"{i+1} fold : ")
                yield trn_idx, val_idx

        elif self.split_type == 'skfold':
            skf = StratifiedKFold(n_splits=self.nfolds, shuffle=self.shuffle)
            for i, (trn_idx, val_idx) in enumerate(skf.split(self.df)):
                print(f"{i+1} fold : ")
                yield trn_idx, val_idx



if __name__ == "__main__":
    df = pd.read_csv(config.TRAIN_FILE)
    cv = CrossValidation(df, config.SPLIT_TYPE, config.SEED, config.NFOLDS, config.SHUFFLE)
    
    for fold, (trn_df, val_df) in enumerate(cv.split()):
        print(fold)
        print(trn_df.head())