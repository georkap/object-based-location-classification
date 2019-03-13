# -*- coding: utf-8 -*-
"""
Contains the classes for ANN and LSTM model stubs

@author: Georgios Kapidis
"""

import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims, dropout=0.0):
        '''
        Params: 
            input_size - (int) - number of input features 
            output_size - (int) - number of output features
            hidden_dims - 1d enumerable of ints e.g. list of ints - number of hidden units per layer
            dropout - (float) - value for dropout in range [0-1], default is 0.0
        '''
        super(ANN, self).__init__() # instantiate class as a torch nn module
        self.linear_layers = nn.ModuleList() # list of layers  
        self.activation = nn.ReLU(inplace=True) # activation
        for i, dim in enumerate(hidden_dims): # create the hidden layers 
            if i == 0:
                self.linear_layers.append(nn.Linear(input_size, dim))
            else:
                self.linear_layers.append(nn.Linear(hidden_dims[i-1], dim))
        self.dropout = nn.Dropout(p=dropout) # add the dropout layer
        self.output = nn.Linear(self.linear_layers[-1].out_features, output_size) # create the output layer
        
        
    def forward(self, x):
        '''
        Params:
            x - (torch.Tensor) - input to the network with shape [batch, input_size]
        '''
        for linear in self.linear_layers:
            x = self.activation(linear(x))
        x = self.dropout(x)
        out = self.output(x)
        return out
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        '''
        Params: 
            input_size - (int) - number of input features 
            hidden_size - (int) - number of hidden units for every lstm cell
            num_layers - (int) - number of stacked lstm layers
            num_classes - (int) - number of output classes
            dropout - (float) - value for dropout in range [0-1], default is 0.0
        '''
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True, 
                            batch_first=False, dropout=0, bidirectional=False) # create lstm layers
        
        self.dropout = nn.Dropout(p=dropout) # add the dropout layer
        self.fc = nn.Linear(hidden_size, num_classes) # create the output layer
        
    def forward(self, seq_batch_vec):
        '''
        Params:
            seq_batch_vec - (torch.Tensor) - input the network with shape [sequence, batch, input_size]
        Note the lstm expects sequence as dim 2, batch as dim 1 and the features as dim 0
        '''
        # wont use padding as I will be keeping a fixed sequence size
        sequence_size = seq_batch_vec.size(0)
        batch_size = seq_batch_vec.size(1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        
        # forward the input through the lstm 
        lstm_out, (hn, cn) = self.lstm(seq_batch_vec, (h0, c0))
        lstm_out = self.dropout(lstm_out) # apply dropout
        
        # get an output for the final sequence step of every sequence in the batch
        output = self.fc(lstm_out[sequence_size-1, list(range(batch_size)), :])
        
        return output