import pandas as pd
from config import *
import string

if __name__ == "__main__":
    df = pd.read_csv('../output/submission/output.csv')
    print(df.head(10))
    