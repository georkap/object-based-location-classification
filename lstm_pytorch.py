# -*- coding: utf-8 -*-
"""
Script for training and testing classification of locations based on the detected objects in the adl dataset
utilizing temporal sequences of objects with LSTM

tested on pytorch 0.4.0
@author: Georgios Kapidis
"""

import time
import torch
import torch.utils.data
from utils import models, dataloaders, other

def train_lstm(model, optimizer, criterion, train_iterator, cur_epoch, log_file, lr_scheduler):
    batch_time, losses, top1, top5 = other.AverageMeter(), other.AverageMeter(), other.AverageMeter(), other.AverageMeter()
    model.train()
    
    lr_scheduler.step()
    
    other.print_and_save('*********', log_file)
    other.print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_iterator):
        
        inputs = torch.tensor(inputs, requires_grad=True).cuda()
        targets = torch.tensor(targets).cuda()

        inputs = inputs.transpose(1, 0)
        output = model(inputs)

        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t1, t5 = other.accuracy(output.detach().cpu(), targets.cpu(), topk=(1,5))
        top1.update(t1.item(), output.size(0))
        top5.update(t5.item(), output.size(0))
        losses.update(loss.item(), output.size(0))
        batch_time.update(time.time() - t0)
        t0 = time.time()
        other.print_and_save('[Epoch:{}, Batch {}/{} in {:.3f} s][Loss {:.4f}[avg:{:.4f}], Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]], LR {:.6f}'.format(
                    cur_epoch, batch_idx, len(train_iterator), batch_time.val, losses.val, losses.avg, top1.val, top1.avg, top5.val, top5.avg, 
                    lr_scheduler.get_lr()[0]), log_file)

def test_lstm(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = other.AverageMeter(), other.AverageMeter(), other.AverageMeter()
    with torch.no_grad():
        model.eval()
        other.print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            inputs = inputs.transpose(1, 0)
            output = model(inputs)
            
#            loss = criterion(output, targets)
            loss = torch.zeros(1).cuda()

            for k in range(targets.size(1)):
                t1, t5 = other.accuracy(output.detach().cpu(), targets[:,k].detach().cpu(), topk=(1,5))
                top1.update(t1.item(), output.size(0))
                top5.update(t5.item(), output.size(0))                
            losses.update(loss.item(), output.size(0))

            other.print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        other.print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg

def run_model(num_input, num_layers, sequence_size, batch, dropout,
              epochs, case, hot, lr_base, train_dir, test_dir):
    save_all_weights = False
    num_hidden = 2*num_input
    model_name = 'lstm_{}_{}_{}_{}_{}_{}_{}_sgd_{}'.format(num_input, sequence_size, batch, str(dropout).split('.')[0]+str(dropout).split('.')[1], epochs, case, hot, str(lr_base).split('.')[0]+str(lr_base).split('.')[1])
    output_dir, log_file = other.init_folders("outputs", model_name, False, True)
    
    model_ft = models.LSTM(num_input, num_hidden, num_layers, 7, dropout)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    other.print_and_save("Model loaded to gpu", log_file)
    train_loader = dataloaders.DataLoaderLSTM(train_dir, num_input, sequence_size, is_training=True)
    test_loader = dataloaders.DataLoaderLSTM(test_dir, num_input, sequence_size, is_training=False)
    train_iterator = torch.utils.data.DataLoader(train_loader, batch_size=batch,
                                                 shuffle=True, num_workers=0,
                                                 pin_memory=True)
    test_iterator = torch.utils.data.DataLoader(test_loader, batch_size=batch, num_workers=0,
                                                pin_memory=True)
    
    params_to_update = model_ft.parameters()
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            other.print_and_save("\t{}".format(name), log_file)    
    
    optimizer = torch.optim.SGD(params_to_update, lr=lr_base, momentum=0.9, weight_decay=0.0005)
    #optimizer = torch.optim.Adam(params_to_update, lr=lr_base, weight_decay=0.0005)
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1)
    
    new_top1, top1 = 0.0, 0.0
    for epoch in range(epochs):
        train_lstm(model_ft, optimizer, ce_loss, train_iterator, epoch, log_file, lr_scheduler)
        new_top1 = test_lstm(model_ft, ce_loss, test_iterator, epoch, "Test", log_file)
        top1 = other.save_checkpoints(model_ft, optimizer, top1, new_top1,
                                      save_all_weights, output_dir, model_name, epoch,
                                      log_file)

#%% TRAINING BELOW
batch=16
dropout=0.0
epochs=150
lr_base = 0.01
num_layers = 2
sequence_size = 20
hot = 'oh'
threshold = 3
num_input = 20
dataset = 'adl20'

dir_oh_l = r"_data/one_hot_7_classes/labels/one_hot_{}_labels_locations/{}"
dir_mh_l = r"_data/multi_hot_classification/from_labels/multi_hot_7_classes/multi_hot_{}_labels_locations_sampled/{}"
dir_oh_d = r"_data/one_hot_7_classes/{}/one_hot_{}_detections_locations_sampled/{}"
dir_mh_d = r"_data/multi_hot_classification/{}/multi_hot_7_classes/multi_hot_{}_detections_locations_sampled/{}"
        
 
case = "d2d_0{}".format(threshold)
_dir = dir_oh_d if hot == 'oh' else dir_mh_d
train_dir = _dir.format(threshold, dataset, 'train')
test_dir = _dir.format(threshold, dataset, 'test')

run_model(num_input, num_layers, sequence_size, batch, dropout,
          epochs, case, hot, lr_base, train_dir, test_dir)
