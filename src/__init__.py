import pandas as pd

if __name__ == "__main__":
    #df = pd.read_csv('../input/train.csv')
    df = pd.read_csv('../input/test.csv')
    print(df[df['textID'] == 'f87dea47db'])
    print(df.head())
    print(df.keys())
    