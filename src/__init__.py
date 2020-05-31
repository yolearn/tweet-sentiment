import pandas as pd
from config import *
import string

if __name__ == "__main__":
    df = pd.read_csv('../input/train.csv')
    print(len(df))
    df.dropna().to_csv('../input/train.csv', index=False)
    
    df = pd.read_csv('../input/train.csv')
    print(len(df))