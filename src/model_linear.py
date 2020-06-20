import torch
import transformers
from transformers import *
import torch.nn as nn
import torch.nn.functional as F
import time
import config

#model = RobertUncaseQa(MODEL_PATH, MODEL_CONF, args['EMBEDDING_SIZE'], 2, args['CNN_OUTPUT_CHANNEL'], args['CNN_KERNEL_WIDTH']).to(args['DEVICE'])

class RobertUncaseQa(transformers.BertPreTrainedModel):
    def __init__(self, robert_path, conf, embedding_size=768, cnn_output_channel=1, kernel_width=1, dropout_rate=0.1, last_layer=3, target_size=2):
        super(RobertUncaseQa, self).__init__(conf)
        self.robert_path = robert_path
        self.model = transformers.RobertaModel.from_pretrained(self.robert_path, config=conf)
        self.kernel_width = kernel_width
        self.cnn_output_channel = cnn_output_channel
        self.target_size = target_size
        self.last_layer = last_layer
        self.embedding_size = embedding_size 
        self.kernel_size = (self.kernel_width, self.embedding_size * self.last_layer)
        self.padding = 0
        self.linear1 = nn.Linear(self.embedding_size * self.last_layer, self.target_size)
        self.st_conv2d = nn.Conv2d(1, self.cnn_output_channel, self.kernel_size)
        self.end_conv2d = nn.Conv2d(1, self.cnn_output_channel, self.kernel_size)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, ids, mask_id, token_type_id):
        """
        sequence_output : (batch_size, max_len, embedding_size)
        pooled_output : (batch, embedding_size)
        hidden_layer : tuple
        """
        sequence_output, pooled_outputm, hidden_layer = self.model(ids, mask_id, token_type_id)
        
        sequence_output = torch.cat([hidden_layer[i*-1] for i in range(self.last_layer, 0, -1)], dim=-1)
        
        logits = self.linear1(sequence_output)
        start_logit, end_logit = torch.split(logits, 1, dim=-1)
        start_logit = start_logit.squeeze(-1)
        end_logit = end_logit.squeeze(-1)

        return start_logit, end_logit


class AlbertQa(transformers.BertPreTrainedModel):
    def __init__(self, robert_path, conf, embedding_size=768, cnn_output_channel=1, kernel_width=1, dropout_rate=0.1, last_layer=3, target_size=2):
        super(AlbertQa, self).__init__(conf)
        self.robert_path = robert_path
        self.model = transformers.RobertaModel.from_pretrained(self.bert_path, config=conf)
        self.embedding_size = embedding_size
        self.target_size = target_size
        self.linear1 = nn.Linear(self.embedding_size, target_size)
        self.dropout = nn.Dropout(0.5)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear1.bias, 0)

    def forward(self, ids, mask_id, token_type_id):
        """
        sequence_output : (batch_size, max_len, embedding_size)
        pooled_output : (batch, embedding_size)
        hidden_layer : tuple
        """
        sequence_output, pooled_outputm, hidden_layer = self.model(ids, mask_id, token_type_id)
        
        sequence_output = torch.cat([hidden_layer[i*-1] for i in range(self.last_layer, 0, -1)], dim=-1)
        
        batch_size = sequence_output.size(0)
        max_len = sequence_output.size(1)
        sequence_output = sequence_output.unsqueeze(1)
        
        #x = self.bert_drop(sequence_output)
        #st_x = F.pad(sequence_output, (0, 0, 0, self.padding))
        st_x = self.dropout(sequence_output)
        st_x = self.st_conv2d(st_x)
        st_x = st_x.view(batch_size, max_len)

        #end_x = F.pad(sequence_output, (0, 0, self.padding, 0))
        end_x = self.dropout(sequence_output)
        end_x = self.end_conv2d(end_x)
        end_x = end_x.view(batch_size, max_len)

        return st_x, end_x



class BertUncasedQa(transformers.BertPreTrainedModel):
    def __init__(self, bert_path, conf, embedding_size=768, output_size=2):
        super(BertUncasedQa, self).__init__(conf)
        self.bert_path = bert_path
        self.model = transformers.RobertaModel.from_pretrained(self.bert_path, config=conf)
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.linear1 = nn.Linear(self.embedding_size, output_size)
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear1.bias, 0)
        

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
        # start_logit = self.softmax(start_logit)
        # end_logit = self.softmax(end_logit)

        # class_logit = self.linear2(sequence_output)
        # class_logit = class_logit.view(class_logit.size(0), -1)
        # class_logit = self.linear3(class_logit)
        # class_logit = class_logit.squeeze(-1)
        
        return F.log_softmax(start_logit, dim=1), F.log_softmax(end_logit, dim=1)