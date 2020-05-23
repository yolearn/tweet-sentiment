import torch.nn as nn
import torch
import numpy as np
import string
from utils import jaccard, AverageMeter, cal_jaccard, load_model, cal_accucary
from tqdm import tqdm
import config
import time 
import time
from scipy.special import softmax



def KSLoss(preds, target):
    pred_cdf = torch.cumsum(torch.softmax(preds, dim=1), dim=1)
    target_cdf = torch.cumsum(target, dim=1)
    error = (target_cdf - pred_cdf)**2
    return torch.mean(error)

def custLoss(preds, target):
    pred = torch.argmax(torch.softmax(preds, axis=1), axis=1)
    target = torch.argmax(target, axis=1)
    return torch.sum((target-pred)**2)

def EntropyLoss(pred, target):
    loss = nn.CrossEntropyLoss()(pred,target)

    return loss

def BcwLoss(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)

    return l1 + l2

def loss_fn(*loss):
    return sum(loss)


def trn_loop_fn(data_loader, model, optimzer, device):
    model.train()
    tk = tqdm(data_loader, total=len(data_loader))
    losses = AverageMeter()

    for i, d in enumerate(tk):
        ids = d['ids']
        mask_ids = d['mask_ids']
        token_type_ids = d['token_type_ids']
        target_start_idx = d['target_start_idx']
        target_end_idx = d['target_end_idx']
        offsets = d['offsets']
        orig_sentiment = d['orig_sentiment']
        orig_sele_text = d['orig_sele_text']
        orig_text = d['orig_text']
        targ_sentiment = d['targ_sentiment']

        ids = ids.to(device, dtype=torch.long)
        mask_ids = mask_ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        target_start_idx = target_start_idx.to(device, dtype=torch.float)
        target_end_idx = target_end_idx.to(device, dtype=torch.float)
        targ_sentiment = targ_sentiment.to(device, dtype=torch.long)
        

        optimzer.zero_grad()
        o1, o2, o3 = model(ids, mask_ids, token_type_ids)
        bcw_loss = BcwLoss(o1, o2, target_start_idx, target_end_idx)
        #entropy_loss = EntropyLoss(o3, targ_sentiment)
        loss = loss_fn(bcw_loss)

        loss.backward()
        optimzer.step()
        losses.update(loss.item(), ids.size(0))
        tk.set_postfix(loss=losses.avg)
    
def eval_loop_fn(data_loader, model, device, df_type):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_output_sentiment = []
    fin_offset = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_orig_text = []
    fin_targ_sentiment = []

    for bi, d in enumerate(data_loader):
        ids = d['ids']
        mask_ids = d['mask_ids']
        token_type_ids = d['token_type_ids']
        target_start_idx = d['target_start_idx']
        target_end_idx = d['target_end_idx']
        offsets = d['offsets']
        orig_sentiment = d['orig_sentiment']
        orig_sele_text = d['orig_sele_text']
        orig_text = d['orig_text']
        targ_sentiment = d['targ_sentiment']

        with torch.no_grad():
            ids = ids.to(device, dtype=torch.long)
            mask_ids = mask_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            target_start_idx = target_start_idx.to(device, dtype=torch.float)
            target_end_idx = target_end_idx.to(device, dtype=torch.float)
            targ_sentiment = targ_sentiment.to(device, dtype=torch.long)

            o1, o2, o3= model(ids, mask_ids, token_type_ids)
            fin_output_start.append(torch.softmax(o1, axis=1).cpu().detach().numpy())
            fin_output_end.append(torch.softmax(o2, axis=1).cpu().detach().numpy())
            fin_output_sentiment.append(torch.softmax(o3, axis=1).cpu().detach().numpy())

            fin_offset.append(offsets)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selected.extend(orig_sele_text)
            fin_orig_text.extend(orig_text)
            fin_targ_sentiment.append(targ_sentiment.cpu().detach().numpy())
            
    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)
    fin_offset = np.vstack(fin_offset)
    fin_output_sentiment = np.vstack(fin_output_sentiment)
    fin_targ_sentiment = np.concatenate(fin_targ_sentiment)
    

    #(fin_output_sentiment, fin_targ_sentiment)

    if df_type == 'val' or df_type == 'trn':
        jaccard_score = cal_jaccard(fin_output_start, fin_output_end, fin_offset, fin_orig_sentiment, fin_orig_selected, fin_orig_text)
        return jaccard_score, fin_output_start, fin_output_end


def pred_loop_fn(data_loader, device):
    
    fin_output_start = []
    fin_output_end = []
    fin_offset = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_orig_text = []

    tk = tqdm(data_loader, total=len(data_loader))
    fin_output_string = []

    for bi, d in enumerate(tk):
        ids = d['ids']
        mask_ids = d['mask_ids']
        token_type_ids = d['token_type_ids']
        offsets = d['offsets']
        orig_sentiment = d['orig_sentiment']
        orig_text = d['orig_text']
        
        with torch.no_grad():
            ids = ids.to(device, dtype=torch.long)
            mask_ids = mask_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            
            model1, model2, model3, model4, model5 = load_model()
            output_start1, output_end1 = model1(ids, mask_ids, token_type_ids)
            output_start2, output_end2 = model2(ids, mask_ids, token_type_ids)
            output_start3, output_end3 = model3(ids, mask_ids, token_type_ids)
            output_start4, output_end4 = model4(ids, mask_ids, token_type_ids)
            output_start5, output_end5 = model5(ids, mask_ids, token_type_ids)
        
            output_start = output_start1 + output_start2 + output_start3 + output_start4 + output_start5
            output_end = output_end1 + output_end2 + output_end3 + output_end4 + output_end5
            
            output_start = softmax(output_start.cpu().detach().numpy(), axis=1)
            output_end = softmax(output_end.cpu().detach().numpy(), axis=1)
            
            fin_output_start.append(output_start)
            fin_output_end.append(output_end)
            fin_offset.append(offsets)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_text.extend(orig_text)

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)
    fin_offset = np.vstack(fin_offset)

    for i in range(fin_output_start.shape[0]):
        output_start = fin_output_start[i]
        output_start = np.argmax(output_start)
        output_end = fin_output_end[i]
        output_end = np.argmax(output_end)
        offset = fin_offset[i]
        orig_sentiment = fin_orig_sentiment[i]
        orig_text = fin_orig_text[i]
    
        if output_start > output_end:
            output_end = output_start
        
        output_string = ""
        for j in range(output_start, output_end+1):
            output_string += orig_text[offset[j][0]:offset[j][1]]
            #if (ix+1) < len(offset) and offset[ix][1] < offset[ix+1][0]:
            #    final_output += " "

        if orig_sentiment == 'neutral':
            output_string = orig_text

        fin_output_string.append(output_string)

    return fin_output_string
        