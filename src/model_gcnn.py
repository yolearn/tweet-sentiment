import torch
import transformers
from transformers import *
import torch.nn as nn
import torch.nn.functional as F
import time
import config

class RobertUncaseQa(transformers.BertPreTrainedModel):
    def __init__(self, robert_path, conf, embedding_size=768, cnn_output_channel=32, kernel_width=1, dropout_rate=0.1, last_layer=3, target_size=2):
        super(RobertUncaseQa, self).__init__(conf)
        self.robert_path = robert_path
        self.model = transformers.RobertaModel.from_pretrained(self.robert_path, config=conf)
        self.embedding_size = embedding_size
        self.kernel_width = kernel_width
        self.cnn_output_channel = cnn_output_channel
        self.target_size = target_size
        self.last_layer = last_layer
        self.dropout_rate = dropout_rate
        self.kernel_size = (self.kernel_width, self.embedding_size * self.last_layer)
        self.padding = int((self.kernel_width - 1) / 2)
        
        layers = 10

        self.conv_st_0 = nn.Conv2d(1, self.cnn_output_channel, self.kernel_size, padding=(self.padding, 0))
        self.conv_gate_st_0 = nn.Conv2d(1, self.cnn_output_channel, self.kernel_size, padding=(self.padding, 0))
        self.b_st_0 = nn.Parameter(torch.randn(1, self.cnn_output_channel, self.embedding_size, 1))
        self.c_st_0 = nn.Parameter(torch.randn(1, self.cnn_output_channel, self.embedding_size, 1))

        self.conv_end_0 = nn.Conv2d(1, self.cnn_output_channel, self.kernel_size, padding=(self.padding, 0))
        self.conv_gate_end_0 = nn.Conv2d(1, self.cnn_output_channel, self.kernel_size, padding=(self.padding, 0))
        self.b_end_0 = nn.Parameter(torch.randn(1, self.cnn_output_channel, self.embedding_size, 1))
        self.c_end_0 = nn.Parameter(torch.randn(1, self.cnn_output_channel, self.embedding_size, 1))

        self.fc = nn.Linear(self.cnn_output_channel * self.cnn_output_channel, self.target_size)
        
        self.start_conv_layers = nn.ModuleList(nn.Conv2d(self.cnn_output_channel, self.cnn_output_channel, (self.cnn_output_channel, 1), padding=(self.padding, 0)) for _ in range(layers))
        self.start_conv_gate_layers = nn.ModuleList(nn.Conv2d(self.cnn_output_channel, self.cnn_output_channel, (self.cnn_output_channel, 1), padding=(self.padding, 0)) for _ in range(layers))

        self.end_conv_layers = nn.ModuleList(nn.Conv2d(self.cnn_output_channel, self.cnn_output_channel, (self.cnn_output_channel, 1), padding=(self.padding, 0)) for _ in range(layers))
        self.end_conv_gate_layers = nn.ModuleList(nn.Conv2d(self.cnn_output_channel, self.cnn_output_channel, (self.cnn_output_channel, 1), padding=(self.padding, 0)) for _ in range(layers))
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linear_st = nn.Linear(self.embedding_size * self.last_layer, 1)
        self.linear_end = nn.Linear(self.embedding_size * self.last_layer, 1)

    def forward(self, ids, mask_id, token_type_id):
        """
        sequence_output : (batch_size, max_len, embedding_size)
        pooled_output : (batch, embedding_size)
        """
        sequence_output, pooled_output, last_hidder = self.model(ids, mask_id, token_type_id)
        sequence_output = torch.cat([last_hidder[i*-1] for i in range(self.last_layer, 0, -1)], dim=-1)
        x = sequence_output.unsqueeze(1)
        
        
        st_a = self.conv_st_0(x)
        
        st_b = self.conv_gate_st_0(x)
        st_h = st_a * F.sigmoid(st_b) 

        #st_h = st_a
        #st_res_input = st_h
    
        # for i, (conv, conv_gate) in enumerate(zip(self.start_conv_layers, self.start_conv_gate_layers)):
        #     st_a = conv(st_h)
        #     st_a = st_h
        #     #st_b = conv_gate(st_h)
        #     #st_h = st_a * F.sigmoid(st_b)

        #     if i%2 == 0 :
        #         st_h += st_res_input
        #         st_res_input = st_h
        
        st_h = st_h.squeeze(1)
        st_h = st_h.squeeze(-1)

        end_a = self.conv_end_0(x) 
        end_b = self.conv_gate_st_0(x)
        end_h = end_a * F.sigmoid(end_b)
        #end_res_input = end_h

        # for i, (conv, conv_gate) in enumerate(zip(self.end_conv_layers, self.end_conv_gate_layers)):
        #     end_a = conv(end_h)
        #     end_a = end_h
        #     #end_b = conv_gate(end_h)
        #     #end_h = st_a * F.sigmoid(end_b)
        
        #     if i%2 == 0:
        #         end_h += end_res_input
        #         end_res_input = end_h

        end_h = end_h.squeeze(1)
        end_h = end_h.squeeze(-1)       

        # logits = self.fc(h.view(h.size(0), -1))
        
        # start_logit, end_logit = torch.split(logits, 1, dim=-1)

        return st_h, end_h

