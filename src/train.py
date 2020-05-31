import argparse
from engine import trn_loop_fn, eval_loop_fn
import pandas as pd 
import numpy as np
import torch
from dataload import TweetDataset
from sklearn import model_selection
from model import BertUncasedQa, RobertUncaseQa, AlbertQa
from transformers import AdamW
import os

from cross_val import CrossValidation
import torch.nn as nn
from utils import upload_to_aws, EarlyStopping, set_seed, clean_text, str2bool, pre_process
import time
import transformers
import tokenizers
from torch.optim import lr_scheduler
def run(cv):
    score = []
    start_preds = np.zeros((df.shape[0], args['MAX_LEN']))
    end_preds = np.zeros((df.shape[0], args['MAX_LEN']))
    
    for fold, (trn_idx, val_idx) in enumerate(cv.split()):
        trn_df = df.iloc[trn_idx]
        if args['PRE_CLEAN']:
            print('cleaning......')
            trn_df = pre_process(trn_df)
        
        val_df = df.iloc[val_idx]
        MODEL_PATH = f'{args["DIR"]}/input/{args["MODEL_VERSION"]}/'
        MODEL_CONF = f'{args["DIR"]}/input/{args["MODEL_VERSION"]}/config.json'
        print(f"Initial ....{args['MODEL_VERSION']}")

        if args['MODEL_VERSION'] in ['bert-base-uncased']:
            TOKENIZER = tokenizers.BertWordPieceTokenizer(
                            f"{args['DIR']}/input/vocab.txt",
                            lowercase=True,
                            add_special_tokens=False
                        )
            MODEL_CONF = transformers.AlbertConfig.from_pretrained(MODEL_CONF)
            model = BertConfig(MODEL_PATH, MODEL_CONF, args['EMBEDDING_SIZE'], args['CNN_OUTPUT_CHANNEL'], args['CNN_KERNEL_SZIE']).to(args['DEVICE'])

        elif args['MODEL_VERSION'] in ['roberta-base', 'roberta-base-squad2', 'roberta-large']:  
            TOKENIZER = tokenizers.ByteLevelBPETokenizer(
                            vocab_file=f"{args['DIR']}/input/roberta-base-squad2/vocab.json",
                            merges_file=f"{args['DIR']}/input/roberta-base-squad2/merges.txt",
                            lowercase=True,
                            add_prefix_space=True
                        )
            MODEL_CONF = transformers.RobertaConfig.from_pretrained(MODEL_CONF)
            MODEL_CONF.output_hidden_states = True
            model = RobertUncaseQa(MODEL_PATH, MODEL_CONF, args['EMBEDDING_SIZE'], 2, args['CNN_OUTPUT_CHANNEL'], args['CNN_KERNEL_SZIE'], args['DROPOUT_RATE']).to(args['DEVICE'])
                        
        elif args['MODEL_VERSION'] in ['albert-base-v2']:
            MODEL_PATH = 'albert-base-v2'
            TOKENIZER = SentencePieceTokenizer(f'{args["DIR"]}/input/spiece.model')
            
            MODEL_CONF = transformers.AlbertConfig.from_pretrained(MODEL_CONF)
            model = AlbertQa(MODEL_PATH, MODEL_CONF, args['EMBEDDING_SIZE'], args['CNN_OUTPUT_CHANNEL'], args['CNN_KERNEL_SZIE']).to(args['DEVICE'])

        
        print(MODEL_CONF)
        trn_dataset = TweetDataset(
            text=trn_df['text'].values,
            selected_text=trn_df['selected_text'].values,
            sentiment=trn_df['sentiment'].values,
            tokenizer=TOKENIZER,
            max_len=args['MAX_LEN'],
            model_type=args['MODEL_VERSION']
        )

        val_dataset = TweetDataset(
            text=val_df['text'].values,
            selected_text=val_df['selected_text'].values,
            sentiment=val_df['sentiment'].values,
            tokenizer=TOKENIZER,
            max_len=args['MAX_LEN'],
            model_type=args['MODEL_VERSION']
        )

        trn_data_loader = torch.utils.data.DataLoader(
            dataset = trn_dataset, 
            batch_size = args['BATCH_SIZE']
        )   

        val_data_loader = torch.utils.data.DataLoader(
            dataset = val_dataset, 
            batch_size = args['BATCH_SIZE']
        ) 


        model_pth = f'{args["DIR"]}/model/{args["MODEL_NAME"]}/fold{fold+1}.pth'
        earlystop = EarlyStopping(path=model_pth, patience=args['PATIENCE'])
        
        model = nn.DataParallel(model)
        optimizer = AdamW(model.parameters(), lr=args['LR'])
        #scheduler = ReduceLROnPlateau(optimizer, 'min')

        for i in range(args['EPOCH']):       
            trn_loop_fn(trn_data_loader, model, optimizer, args['DEVICE'])
            #cur_score = eval_loop_fn(trn_data_loader, model, config.DEVICE)
            #print(f"Train {i+1} EPOCH : JACCARDS = {cur_score}")
            cur_score, pred1, pred2 = eval_loop_fn(val_data_loader, model, args['DEVICE'], 'val')
            print(f"Train {i+1} EPOCH : AVG JACCARDS      = {cur_score['avg_score']}")
            #print(f"Train {i+1} EPOCH : AVG ACCURACY      = {accuracy}")
            print(f"Train {i+1} EPOCH : NEUTRAL JACCARDS  = {cur_score['neu_score']}")
            print(f"Train {i+1} EPOCH : POSITIVE ACCARDS  = {cur_score['pos_score']}")
            print(f"Train {i+1} EPOCH : NEGATIVE JACCARDS = {cur_score['neg_score']}")
            

            earlystop(cur_score['avg_score'], model, i+1, pred1, pred2)
            if earlystop.earlystop:
                print("Early stopping")
                break

        score.append(earlystop.max)

        if args['SPLIT_TYPE'] != 'pure_split':
            start_preds[val_idx] = earlystop.pred1
            end_preds[val_idx] = earlystop.pred2
    
    text_file = open(f'{args["DIR"]}/model/{args["MODEL_NAME"]}/score.txt', "w")
    text_file.write(str(score))
    text_file.write("\n")
    text_file.write(str(sum(score) / len(score)))
    text_file.close()


    if args['SPLIT_TYPE'] != 'pure_split':
        cv_preds = np.concatenate([start_preds, end_preds], axis=1)
        pd.DataFrame(cv_preds).to_csv(f'{args["DIR"]}/model/{args["MODEL_NAME"]}/cv.csv', index=False)   

if __name__ == '__main__':
    #CUDA_VISIBLE_DEVICES=1 python3 train.py
    parser = argparse.ArgumentParser(description="Let's tuning hyperparameter")
    parser.add_argument('--NFOLDS', default=5, type=int)
    parser.add_argument('--SPLIT_TYPE', default='kfold')         #kfold, pure_split
    #parser.add_argument('--TRAIN_NUM', default=1000, type=int)
    parser.add_argument('--DIR', default="..")
    parser.add_argument('--SEED', default=42, type=int)
    parser.add_argument('--DEVICE', default=torch.device('cuda'))   #cpu          
    parser.add_argument('--SHUFFLE', default=True)
    parser.add_argument('--PRE_CLEAN', default=True)
    parser.add_argument('--MODEL_NAME', default='baseline')
    
    #MODLE HYPER PARAMETER
    parser.add_argument('--EMBEDDING_SIZE', default=768, type=int)
    parser.add_argument('--CNN_KERNEL_SZIE', default=1, type=int)
    parser.add_argument('--CNN_OUTPUT_CHANNEL', default=1, type=int)
    parser.add_argument('--BATCH_SIZE', default=32, type=int)
    parser.add_argument('--MODEL_VERSION', default='roberta-base')
    parser.add_argument('--MAX_LEN', default=128, type=int)
    parser.add_argument('--EPOCH', default=3, type=int)
    parser.add_argument('--LR', default=3e-5, type=int)
    parser.add_argument('--PATIENCE', default=1, type=int)
    parser.add_argument('--DROPOUT_RATE', default=0.1, type=int)

    args = parser.parse_args()
    args = dict(vars(args))
    
    for key, value in args.items():
        print(key, value)


    set_seed(args['SEED'])
    df = pd.read_csv(f"{args['DIR']}/input/train.csv").copy(deep=True)
    df['text'] = df['text'].astype(str)
    df['selected_text'] = df['selected_text'].astype(str)
    cv = CrossValidation(df, args['SPLIT_TYPE'], args['SEED'], args['NFOLDS'], args['SHUFFLE'])
    
    if os.path.exists(f"../model/{args['MODEL_NAME']}"):
        pass
    else:
        os.mkdir(f"../model/{args['MODEL_NAME']}")

    run(cv)



    