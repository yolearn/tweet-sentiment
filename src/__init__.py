import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('../input/train.csv')
    print(df.head())
    print(df.keys())
    