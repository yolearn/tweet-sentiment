import torch
import transformers
import torch.nn as nn

class BertUncasedQa(nn.Module):
    def __init__(self, bert_path):
        super(BertUncasedQa, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.linear = nn.Linear(768,2)

    def forward(self, ids, mask, token_type_id):
        #o1 the last hideen state, o2 CLS token
        sequence_output, pooled_output = self.bert(ids, mask, token_type_id)
        logits = self.linear(sequence_output)
        start_logit, end_logit = logits.split(1, dim=-1)
        start_logit = start_logit.squeeze(-1)
        end_logit = end_logit.squeeze(-1)
        
        return start_logit, end_logit