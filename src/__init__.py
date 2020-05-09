import pandas as pd
from config import *
import string

if __name__ == "__main__":
    #df = pd.read_csv('../input/train.csv')
    df = pd.read_csv('../input/test.csv')
    #print(df.head())
    #print(df[df['textID'] == 'eee518ae67']['text'].values[0].split())
    # print(df.head())
    # print(df.keys())

    
    for text in df['text'][:30]:
        print(TOKENIZER.encode(text).tokens)
        print(TOKENIZER.encode(text).ids)
        print(TOKENIZER.encode(text).offsets)
        print('-'*10)
        # output_token = " ".join(output_token)
        # output_token = [token for i, token in enumerate(output_token.split()) if token not in ("[CLS]", "[SEP]")]

        # final_output = ''
        # for token in output_token:
        #     if token.startswith("##"):
        #         final_output+=token[2:]
        #     elif token in string.punctuation :
        #         final_output+=token
        #     else:
        #         final_output+=' '
        #         final_output+=token
        # final_output = final_output.strip()
        # print(text)
        # print(final_output)
        # print(output_token)
        # print('-'*10)