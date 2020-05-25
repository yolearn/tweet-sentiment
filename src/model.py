import torch
import transformers
from transformers import *
import torch.nn as nn
import torch.nn.functional as F
import time
import config

class BertUncasedQa(nn.Module):
    def __init__(self, bert_path):
        super(BertUncasedQa, self).__init__()
        self.bert_path = bert_path
        self.model = transformers.BertModel.from_pretrained(self.bert_path)
        #self.bert_drop = nn.Dropout(config.DROPOUT_RATE)
        self.linear1 = nn.Linear(1024,2)
        self.linear2 = nn.Linear(1024,1)
        self.linear3 = nn.Linear(128,3)

    def forward(self, ids, mask_id, token_type_id):
        #o1 the last hideen state, o2 CLS token
        sequence_output, pooled_output = self.model(ids, mask_id, token_type_id)
        #x = self.bert_drop(sequence_output)
        logits = self.linear1(sequence_output)

        start_logit, end_logit = torch.split(logits, 1, dim=-1)
        start_logit = start_logit.squeeze(-1)
        end_logit = end_logit.squeeze(-1)

        class_logit = self.linear2(sequence_output)
        class_logit = class_logit.view(class_logit.size(0), -1)
        class_logit = self.linear3(class_logit)
        class_logit = class_logit.squeeze(-1)
        
        return start_logit, end_logit, class_logit



class RobertUncaseQa(transformers.BertPreTrainedModel):
    def __init__(self, robert_path, conf):
        super(RobertUncaseQa, self).__init__(conf)
        self.robert_path = robert_path
        self.model = transformers.RobertaModel.from_pretrained(self.robert_path, config=conf)
        #self.bert_drop = nn.Dropout(config.DROPOUT_RATE)
        self.linear1 = nn.Linear(768,2)
        # self.linear2 = nn.Linear(768,1)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear1.bias, 0)
        # self.linear3 = nn.Linear(128,3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ids, mask_id, token_type_id):
        """
        sequence_output : (batch_size, max_len, embedding_size)
        pooled_output : (batch, embedding_size)
        """
        sequence_output, pooled_output = self.model(ids, mask_id, token_type_id)
        #x = self.bert_drop(sequence_output)
        logits = self.linear1(sequence_output)

        start_logit, end_logit = torch.split(logits, 1, dim=-1)
        start_logit = start_logit.squeeze(-1)
        end_logit = end_logit.squeeze(-1)
        start_logit = self.softmax(start_logit)
        end_logit = self.softmax(end_logit)

        # class_logit = self.linear2(sequence_output)
        # class_logit = class_logit.view(class_logit.size(0), -1)
        # class_logit = self.linear3(class_logit)
        # class_logit = class_logit.squeeze(-1)
        
        return start_logit, end_logit


class AlbertQa(nn.Module):
    def __init__(self, albert_path):
        super(AlbertQa, self).__init__()
        self.albert_path = albert_path
        self.model = transformers.AlbertModel.from_pretrained(self.albert_path)
        self.linear = nn.Linear(768,2)


    def forward(self, ids, mask_id, token_type_id):
        sequence_output, pooled_output = self.model(ids, mask_id, token_type_id)

        logits = self.linear(sequence_output)
        start_logit, end_logit = torch.split(logits, 1, dim=-1)
        start_logit = start_logit.squeeze(dim=-1)
        end_logit = end_logit.squeeze(dim=-1)

        return start_logit, end_logit