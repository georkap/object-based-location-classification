# -*- coding: utf-8 -*-
"""
Script for training and testing classification of locations based on the detected objects in the adl dataset

tested on pytorch 0.4.0
@author: Georgios Kapidis
"""

import time
import torch
from utils import models, dataloaders, other

def train_ann(model, optimizer, criterion, train_iterator, cur_epoch, log_file, lr_scheduler=None):
    batch_time, losses, top1, top5 = other.AverageMeter(), other.AverageMeter(), other.AverageMeter(), other.AverageMeter()
    model.train()
    
    lr_scheduler.step()
    
    other.print_and_save('*********', log_file)
    other.print_and_save('Beginning of epoch: {}'.format(cur_epoch), log_file)
    t0 = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_iterator):            
        inputs = torch.tensor(inputs, requires_grad=True).cuda()
        targets = torch.tensor(targets).cuda()
        
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

def test_ann(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = other.AverageMeter(), other.AverageMeter(), other.AverageMeter()
    with torch.no_grad():
        model.eval()
        other.print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
        for batch_idx, (inputs, targets) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            output = model(inputs)
            loss = criterion(output, targets)

            t1, t5 = other.accuracy(output.detach().cpu(), targets.detach().cpu(), topk=(1,5))
            top1.update(t1.item(), output.size(0))
            top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))

            other.print_and_save('[Epoch:{}, Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]'.format(
                        cur_epoch, batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg), log_file)

        other.print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg

batch=64
dropout=0.0
epochs=150

threshold=3
num_input=20
dataset='adl20'
lr_base = 0.01
case = "d2d_0{}".format(threshold)
model_name = 'ann_{}_{}_{}_{}_{}_oh_sgd_{}'.format(num_input, batch, str(dropout).split('.')[0]+str(dropout).split('.')[1], epochs, case, str(lr_base).split('.')[0]+str(lr_base).split('.')[1])
output_dir, log_file = other.init_folders("outputs", model_name, False, True)
        
model_ft = models.ANN(num_input, 7, [64, 256, 128, 64], dropout=dropout)
model_ft = torch.nn.DataParallel(model_ft).cuda()
other.print_and_save("Model loaded to gpu", log_file)

train_dir = r"_data/one_hot_7_classes/{}/one_hot_{}_detections_locations_sampled/train".format(threshold, dataset)
test_dir = r"_data/one_hot_7_classes/{}/one_hot_{}_detections_locations_sampled/test".format(threshold, dataset)

train_loader = dataloaders.DataLoaderANN(train_dir)
test_loader = dataloaders.DataLoaderANN(test_dir)

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
    train_ann(model_ft, optimizer, ce_loss, train_iterator, epoch, log_file, lr_scheduler)
    new_top1 = test_ann(model_ft, ce_loss, test_iterator, epoch, "Test", log_file)
    top1 = other.save_checkpoints(model_ft, optimizer, top1, new_top1,
                                  False, output_dir, model_name, epoch,
                                  log_file)       
