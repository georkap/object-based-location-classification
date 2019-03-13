# -*- coding: utf-8 -*-
"""
Other utilities useful to both ANN and LSTM location classification schemes

@author: Georgios Kapidis
"""

import os, sys, shutil, torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    if maxk > output.shape[1]:
        maxk = output.shape[1]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def init_folders(base_output_dir, model_name, resume, logging):
    '''Initialize the output folder and log file for the current model'''
    output_dir = os.path.join(base_output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if not resume:
            sys.exit("Attempted to overwrite previous folder, exiting..")
    
    log_file = os.path.join(base_output_dir, model_name, model_name+".txt") if logging else None
    return output_dir, log_file

def print_and_save(text, path):
    print(text)
    if path is not None:
        with open(path, 'a') as f:
            print(text, file=f)

def save_best_checkpoint(top1, new_top1, output_dir, model_name, weight_file):
    isbest = True if new_top1 >= top1 else False
    if isbest:
        best = os.path.join(output_dir, model_name+'_best.pth')
        shutil.copyfile(weight_file, best)
        top1 = new_top1
    return top1

def save_checkpoints(model_ft, optimizer, top1, new_top1, save_all_weights, 
                     output_dir, model_name, epoch, log_file):
    ''' Save best and current epoch checkpoints for a model. Can choose to save all epochs with save_all_weight=True'''
    if save_all_weights:
        weight_file = os.path.join(output_dir, model_name + '_{:03d}.pth'.format(epoch))
    else:
        weight_file = os.path.join(output_dir, model_name + '_ckpt.pth')
    print_and_save('Saving weights to {}'.format(weight_file), log_file)
    torch.save({'epoch': epoch,
                'state_dict': model_ft.state_dict(),
                'optimizer': optimizer.state_dict(),
                'top1': new_top1}, weight_file)
    top1 = save_best_checkpoint(top1, new_top1, output_dir, model_name, weight_file)
    return top1
