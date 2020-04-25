import torch
import transformers
import torch.nn as nn

class BertUncased():
    def __init__(self, bert_path):
        super(BertUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.linear = nn.Linear(768,2)

    def forward(self, ids, mask, token_type_id):
        #o1 the last hideen state, o2 CLS token
        o1, o2 = self.bert(ids, mask, token_type_id)
        logit = self.linear(o2)
        start_logit, end_logit = logit.split(1, dim=-1)
        start_logit = start_logit.squeeze(1)
        end_logit = end_logit.squeeze(1)
        
        return start_logit, end_logit