import torch.nn as nn
import torch
import numpy as np
import string
from utils import jaccard, AverageMeter, cal_jaccard, load_model
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

def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)

    return l1+l2

"""
def trn_loop_fn(data_loader, model, optimzer, device):
    model.train()
    losses = AverageMeter()
    tk = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk):
        ids = d['ids']
        mask_ids = d['mask_ids']
        token_type_ids = d['token_type_ids']
        target_start_idx = d['target_start_idx']
        target_end_idx = d['target_end_idx']

        ids = ids.to(device, dtype=torch.long)
        mask_ids = mask_ids.to(device, dtype=torch.long)
        token_type_ids  = token_type_ids.to(device, dtype=torch.long)
        target_start_idx = target_start_idx.to(device, dtype=torch.float)
        target_end_idx = target_end_idx.to(device, dtype=torch.float)

        optimzer.zero_grad()
        o1, o2 = model(ids, mask_ids, token_type_ids)
        loss = loss_fn(o1, o2, target_start_idx, target_end_idx)
        loss.backward()
        optimzer.step()      

        losses.update(loss.item(), ids.size(0))
        tk.set_postfix(loss=losses.avg)

def eval_loop_fn(data_loader, model, device):
    model.eval()
    fin_output_start = []
    fin_output_end = []
    fin_orig_sentiment = []
    #fin_padding_len = []
    #fin_text_token = []
    fin_origin_text = []
    fin_orig_selected = []
    fin_offset = []
    
    for bi, d in enumerate(data_loader):
        ids = d['ids']
        mask_ids = d['mask_ids']
        token_type_ids = d['token_type_ids']
        targets_start = d['target_start_idx']
        targets_end = d['target_end_idx']
        orig_sentiment = d['orig_sentiment']
        orig_sele_text = d['orig_sele_text']
        orig_text = d['orig_text']
        offsets = d['offsets'].numpy()
        fin_offset.append(offsets)

        with torch.no_grad():
            ids = ids.to(device, dtype=torch.long)
            mask_ids = mask_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.float)
            targets_end = targets_end.to(device, dtype=torch.float)

            output_start, output_end = model(ids, mask_ids, token_type_ids)
            loss = loss_fn(output_start, output_end, targets_start, targets_end)

            fin_output_start.append(torch.softmax(output_start, axis=1).cpu().detach().numpy())
            fin_output_end.append(torch.softmax(output_end, axis=1).cpu().detach().numpy())
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selected.extend(orig_sele_text)
            fin_origin_text.extend(orig_text)

    fin_offset = np.vstack(fin_offset)
    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    jac_score = []
    for j in range(len(fin_output_start)):
        origin_selected = fin_orig_selected[j]
        orig_sentiment = fin_orig_sentiment[j]
        orig_text = fin_origin_text[j]
        offset = fin_offset[j]
        start_idx = fin_output_start[j]
        end_idx = fin_output_end[j]
        start_idx = np.argmax(start_idx)
        end_idx = np.argmax(end_idx)
        # start_idx = np.nonzero(start_idx)[0][0]
        # end_idx = np.nonzero(end_idx)[0][0]

        if end_idx < start_idx:
            end_idx = start_idx
        
        final_output = ""
        count = 0
        for ix in range(start_idx, end_idx+1):
            final_output += orig_text[offset[ix][0]:offset[ix][1]]     
            if (ix+1) < len(offset) and offset[ix][1] < offset[ix+1][0]:
                print('出事情囉')
                final_output += " "
            
        # if orig_sentiment == 'neutral' or len(orig_text.split()) < 4:
        #     final_output = orig_text
        
        jac = jaccard(final_output, origin_selected)
        jac_score.append(jac)

    return np.mean(jac_score)
"""


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

        ids = ids.to(device, dtype=torch.long)
        mask_ids = mask_ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        target_start_idx = target_start_idx.to(device, dtype=torch.float)
        target_end_idx = target_end_idx.to(device, dtype=torch.float)

        optimzer.zero_grad()
        o1, o2 = model(ids, mask_ids, token_type_ids)
        loss = loss_fn(o1, o2, target_start_idx, target_end_idx)
        loss.backward()
        optimzer.step()
        losses.update(loss.item(), ids.size(0))
        tk.set_postfix(loss=losses.avg)
    
def eval_loop_fn(data_loader, model, device, df_type):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_offset = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_orig_text = []

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
        
        with torch.no_grad():
            ids = ids.to(device, dtype=torch.long)
            mask_ids = mask_ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            target_start_idx = target_start_idx.to(device, dtype=torch.float)
            target_end_idx = target_end_idx.to(device, dtype=torch.float)

            o1, o2 = model(ids, mask_ids, token_type_ids)
            loss = loss_fn(o1, o2, target_start_idx, target_end_idx)

            fin_output_start.append(torch.softmax(o1, axis=1).cpu().detach().numpy())
            fin_output_end.append(torch.softmax(o2, axis=1).cpu().detach().numpy())
            fin_offset.append(offsets)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selected.extend(orig_sele_text)
            fin_orig_text.extend(orig_text)

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)
    fin_offset = np.vstack(fin_offset)

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
        