import torch
import transformers
import torch.nn as nn

class BertUncasedQa(nn.Module):
    def __init__(self, bert_path):
        super(BertUncasedQa, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(bert_path)
        self.bert_drop = nn.Dropout(0.5)
        self.linear1 = nn.Linear(768,100)
        self.linear2 = nn.Linear(100,2)

    def forward(self, ids, mask, token_type_id):
        #o1 the last hideen state, o2 CLS token
        sequence_output, pooled_output = self.bert(ids, mask, token_type_id)
        x = self.bert_drop(sequence_output)
        x = self.linear1(x)
        x = self.bert_drop(x)
        logits = self.linear2(x)
        start_logit, end_logit = logits.split(1, dim=-1)
        start_logit = start_logit.squeeze(-1)
        end_logit = end_logit.squeeze(-1)
        
        return start_logit, end_logit