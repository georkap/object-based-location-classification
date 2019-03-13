# -*- coding: utf-8 -*-
"""
Contains the data loaders for ANN and LSTM

@author: Georgios Kapidis
"""

import os
import torch.utils.data
import numpy as np

def load_train_file(path):
    '''
    Given a path it loads the binary presence vectors. 
    Target file format:
        space separated zeros and ones: one based location class index
        e.g. 0 1 0 1 ... 0:5
    Note the location classes are always -1 to become zero-based
    '''
    data = []
    labels = []
    with open(path, 'r') as p:
        lines = p.readlines()
        for l in lines:
            datum, label = l.split(":")
            data.append(np.array([np.float32(x) for x in datum.split()]))
            labels.append(np.int64(label)-1)
    return data, labels

class DataLoaderANN(torch.utils.data.Dataset):
    def __init__(self, data_dir, evaluation=False):
        '''
        Given a directory it loads all the files in it into two numpy arrays one for data, one for labels
        self.data:shape=[number of samples, size of binary presence vector]
        self.labels:shape=[number of samples, 1]
        '''
        datafile_names = os.listdir(data_dir)
        datafile_paths = [os.path.join(data_dir, x) for x in datafile_names]
        
        data = []
        labels = []
        for path in  datafile_paths:
            d, l = load_train_file(path)
            data += d
            labels += l
        self.data = np.array(data, dtype=np.float32)
        self.labels = np.array(labels)
        self.evaluation = evaluation
    
    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.data)
    
    def __getitem__(self, index):
        bpv = self.data[index]
        label = self.labels[index]
        
        if self.evaluation:
            return bpv, label, str(index)
        else:
            return bpv, label
    

class DataLoaderLSTM(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_size, sequence_size, is_training=True, evaluation=False):
        '''
        Given a directory it loads all the files in it into two numpy arrays one for data, one for labels
        after splitting them in sequences
        self.data:shape=[number of sequences, sequence_size, input_size]
        self.labels:shape=[number of sequences, sequence_size, 1]
        Params:
            data_dir - (str) - directory containing bpv files
            input_size - (int) - the size of the bpv feature vector
            sequence_size - (int) - size of the sequences
            is_training - (bool) - if True the out_label from the forward function 
                is the majority vote of all the labels in the sequence, 
                if False all the labels are returned in a numpy array, default is True, use False for evaluation
        '''
        datafile_names = os.listdir(data_dir) # load each file
        datafile_paths = [os.path.join(data_dir, x) for x in datafile_names]
        
        sequence_data = []
        sequence_labels = []
        for path in datafile_paths:
            d, l = load_train_file(path) # load the data per file
            assert len(d[0]) == input_size
            num_vectors = len(d)
            # split the data file in sequences separately, 
            # no point in concatenating bpv from different videos
            mod = num_vectors%sequence_size 
            if mod > 0: # append the last sequence with the last value to make it the same sizeas the rest
                for i in range(sequence_size - mod): 
                    d.append(d[-1])
                    l.append(l[-1])
            sd = list(np.array(d, dtype=np.float32).reshape((-1, sequence_size, input_size)))
            sl = list(np.array(l).reshape((-1, sequence_size)))
            sequence_data += sd
            sequence_labels += sl
            
        self.data = np.array(sequence_data, dtype=np.float32)
        self.labels = np.array(sequence_labels)
        self.is_training = is_training
        self.evaluation = evaluation
    
    def __len__(self):
        assert len(self.data) == len(self.labels)
        return len(self.data)
    
    def __getitem__(self, index):
        bpv_seq = self.data[index]
        label = self.labels[index]
        maj_vote = np.bincount(label).argmax()
        
        if self.is_training:
            out_label = maj_vote
        else:
            out_label = label
        
        if self.evaluation:
            return bpv_seq, out_label, str(index)
        else:
            return bpv_seq, out_label
            
        