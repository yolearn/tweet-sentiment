import torch.nn as nn
import torch
import numpy as np
import string
from utils import jaccard, AverageMeter
from tqdm import tqdm

def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss()(o1, t1)
    l2 = nn.BCEWithLogitsLoss()(o2, t2)
    return l1+l2

def trn_loop_fn(data_loader, model, optimzer, device):
    model.train()
    losses = AverageMeter()
    tk = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk):
        ids = d['ids']
        mask_id = d['mask_id']
        token_type_id = d['token_type_id']
        targets_start = d['targets_start']
        targets_end = d['targets_end']

        ids = ids.to(device, dtype=torch.long)
        mask_id = mask_id.to(device, dtype=torch.long)
        token_type_id  = token_type_id.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimzer.zero_grad()
        o1, o2 = model(ids, mask_id, token_type_id)
        loss = loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimzer.step()      
        losses.update(loss.item(), ids.size(0))
        tk.set_postfix(loss=losses.avg)

def eval_loop_fn(data_loader, model, device):
    model.eval()
    fin_output_start = []
    fin_output_end = []
    fin_orig_sentiment = []
    fin_padding_len = []
    fin_text_token = []
    fin_origin_text = []
    fin_orig_selected = []

    for bi, d in enumerate(data_loader):
        ids = d['ids']
        mask_id = d['mask_id']
        token_type_id = d['token_type_id']
        targets_start = d['targets_start']
        targets_end = d['targets_end']
        origin_sentiment = d['origin_sentiment']
        padding_len = d['padding_len']
        text_token = d['text_token']
        selected_text = d['origin_selected_text']
        origin_text = d['origin_text']

        with torch.no_grad():
            ids = ids.to(device, dtype=torch.long)
            mask_id = mask_id.to(device, dtype=torch.long)
            token_type_id = token_type_id.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.float)
            targets_end = targets_end.to(device, dtype=torch.float)

            output_start, output_end = model(ids, mask_id, token_type_id)
            loss = loss_fn(output_start, output_end, targets_start, targets_end)

            fin_output_start.append(torch.sigmoid(output_start).cpu().detach().numpy())
            fin_output_end.append(torch.sigmoid(output_end).cpu().detach().numpy())
            fin_padding_len.extend(padding_len.cpu().detach().numpy().tolist())
            fin_text_token.extend(text_token)
            fin_orig_sentiment.extend(origin_sentiment)
            fin_orig_selected.extend(selected_text)
            fin_origin_text.extend(origin_text)

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)

    jac_score = []

    for j in range(len(fin_output_start)):
        text_token = fin_text_token[j]
        padding_len = fin_padding_len[j]
        origin_selected = fin_orig_selected[j]
        origin_sentiment = fin_orig_sentiment[j]
        origin_text = fin_origin_text[j]

        if padding_len > 0:
            mask_start = fin_output_start[j][:-padding_len] >= THRESHOLD
            mask_end = fin_output_end[j][:-padding_len] >= THRESHOLD
        else:
            mask_start = fin_output_start[j] >= THRESHOLD
            mask_end = fin_output_end[j] >= THRESHOLD

        mask = [0]*len(mask_start)
        idx_start_l = np.nonzero(mask_start)[0]
        idx_end_l = np.nonzero(mask_end)[0]

        if len(idx_start_l) >0:
            idx_start = idx_start_l[0]
            if len(idx_end_l)>0:
                idx_end = idx_end_l[0]
            else:
                idx_end = idx_start 
        else:
            idx_start = idx_end = 0

        for i in range(idx_start, idx_end+1):
            mask[i] = 1

        output_tokens = [token for i, token in enumerate(text_token.split()) if mask[i] == 1]
        output_tokens = [token for token in output_tokens if token not in ("[CLS]", "[SEP]")]

        final_output = ""
        for token in output_tokens:
            if token.startswith("##"):
                final_output+=token[2:]
            elif len(token) == 1 or token in string.punctuation:
                final_output+=token
            else:
                final_output+=' '
                final_output+=token

        final_output = final_output.strip()
        if origin_sentiment == 'neutral' or len(text_token.split()) < 4:
            final_output = origin_text
        
        jac = jaccard(final_output, origin_selected)
        jac_score.append(jac)

    return np.mean(jac_score)

