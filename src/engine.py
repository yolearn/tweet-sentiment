import torch.nn as nn
import torch


def loss_fn(o1, o2, t1, t2):
    l1 = nn.BCEWithLogitsLoss(o1, t1)
    l2 = nn.BCEWithLogitsLoss(o2, t2)
    return l1+l2


def trn_loop_fn(data_loader, model, optimzer, device):
    model.train()
    loss = 0
    for bi, d in enumerate(data_loader):
        token_id = d['token_id']
        mask_id = d['mask_id']
        token_type_id = d['token_type_id']
        targets_start = d['targets_start']
        targets_end = d['targets_end']

        token_id = token_id.to(device, dtype=torch.long)
        mask_id = mask_id.to(device, dtype=torch.long)
        token_type_id  = token_type_id.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        optimzer.zero_grad()
        o1, o2 = model(ids, mask, token_type_id)
        loss += loss_fn(o1, o2, targets_start, targets_end)
        loss.backward()
        optimzer.step()

        

def eval_loop_fn(data_loader, model, device):
    model.eval()
    loss = 0
    for bi, d in enumerate(data_loader):
        token_id = d['token_id']
        mask_id = d['mask_id']
        token_type_id = d['token_type_id']
        targets_start = d['targets_start']
        targets_end = d['targets_end']

        token_id = token_id.to(device, dtype=torch.long)
        mask_id = mask_id.to(device, dtype=torch.long)
        token_type_id  = token_type_id.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.float)
        targets_end = targets_end.to(device, dtype=torch.float)

        o1, o2 = model(ids, mask, token_type_id)
        loss+=loss_fn(o1, o2, targets_start, targets_end)

        