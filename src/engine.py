import torch.nn as nn
import torch
import numpy as np
import string
from utils import jaccard, AverageMeter
from tqdm import tqdm
from config import *
import time 


def loss_fn(o1, o2, t1, t2):
    # l1 = nn.MarginRankingLoss()(o1, t1)
    # l2 = nn.MarginRankingLoss()(o2, t2)
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1+l2


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
            #fin_padding_len.extend(padding_len.cpu().detach().numpy().tolist())
            #fin_text_token.extend(text_token)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selected.extend(orig_sele_text)
            fin_origin_text.extend(orig_text)
            

    fin_offset = np.vstack(fin_offset)
    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    jac_score = []

    for j in range(len(fin_output_start)):
        #text_token = fin_text_token[j]
        #padding_len = fin_padding_len[j]
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
                final_output += " "
            

        # if orig_sentiment == 'neutral' or len(orig_text.split()) < 4:
        #     final_output = orig_text
        
        jac = jaccard(final_output, origin_selected)
        jac_score.append(jac)

    return np.mean(jac_score)



# def trn_loop_fn(data_loader, model, optimzer, device):
#     model.train()
#     optimzer.zero_grad()
#     tk = tqdm(data_loader, len(data_loader))
#     losses = AverageMeter()

#     for i, d enumerate(tk):
#         ids = d['ids']
#         mask_ids = d['mask_ids']
#         token_type_ids = d['token_type_ids']
#         target_start_idx = d['target_start_idx']
#         target_end_idx = d['target_end_idx']
#         offsets = d['offsets']
#         orig_sentiment = d['orig_sentiment']
#         orig_sele_text = d['orig_sele_text']
#         orig_text = d['orig_text']

#         ids = ids.to(device, dtype=torch.long)
#         mask_ids = mask_ids.to(device, dtype=torch.long)
#         token_type_ids = token_type_ids.to(device, dtype=torch.long)
#         target_start_idx = target_start_idx.to(device, dtype=torch.float)
#         target_end_idx = target_end_idx.to(device, dtype=torch.float)

#         o1, o2 = model(ids, token_type_ids, mask_ids)
#         loss = loss_fn(o1, o2, target_start_idx, target_end_idx)
#         loss.backward()
#         optimzer.step()
#         losses.update(loss.item(), ids.size(0))
#         tk.set_postfix(loss=losses.avg)


# def eval_loop_fn(data_loader, model, device):
#     model.eval()

#     fin_output_start = []
#     fin_output_end = []
#     fin_offsets = []
#     fin_
#     with torch.no_grad():
#         for i, d enumerate(data_loader):
#             ids = d['ids']
#             mask_ids = d['mask_ids']
#             token_type_ids = d['token_type_ids']
#             target_start_idx = d['target_start_idx']
#             target_end_idx = d['target_end_idx']
#             offsets = d['offsets']
#             orig_sentiment = d['orig_sentiment']
#             orig_sele_text = d['orig_sele_text']
#             orig_text = d['orig_text']

#             ids = ids.to(device, dtype=torch.long)
#             mask_ids = mask_ids.to(device, dtype=torch.long)
#             token_type_ids = token_type_ids.to(device, dtype=torch.long)
#             target_start_idx = target_start_idx.to(device, dtype=torch.float)
#             target_end_idx = target_end_idx.to(device, dtype=torch.float)

#             o1, o2 = model(ids, token_type_ids, mask_ids) 
#             torch.softmax(o1, dim=1).detach().cpu().numpy()  
#             fin_output_start.append(o1)
#             fin_output_end.append(o2)
#             fin_offsets.append(offsets)
