# -*- coding: utf-8 -*-
"""
Script for evaluation of location classification based on 
the detected objects in the adl dataset
for the LSTM models

output is average f1 scores and recall precision per class

tested on pytorch 0.4.0
@author: Georgios Kapidis
"""

import os
import numpy as np
import torch
import torch.utils.data
from utils import models, dataloaders, other
from sklearn.metrics import f1_score, confusion_matrix

def rec_prec_per_class(confusion_matrix):
    # cm is inversed from the wikipedia example on 3/8/18

    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP
    
    with np.errstate(divide='warn'):
        precision = np.nan_to_num(TP/(TP+FP))
        recall = np.nan_to_num(TP/(TP+FN))
    
    return np.around(100*recall, 2), np.around(100*precision, 2)

def analyze_preds_labels(preds, labels):
    cf = confusion_matrix(labels, preds).astype(int)
    recall, precision = rec_prec_per_class(cf)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)

    with np.errstate(divide='warn'):
        cls_acc = np.around(100*np.nan_to_num(cls_hit / cls_cnt), 2)
    mean_cls_acc = np.mean(cls_acc)
    top1_acc = np.around(100*(np.sum(cls_hit)/np.sum(cf)), 3)
    
    return cf, recall, precision, cls_acc, mean_cls_acc, top1_acc         

def validate_lstm(model, criterion, test_iterator, cur_epoch, dataset, log_file):
    losses, top1, top5 = other.AverageMeter(), other.AverageMeter(), other.AverageMeter()
    outputs = []
    
    other.print_and_save('Evaluating after epoch: {} on {} set'.format(cur_epoch, dataset), log_file)
    with torch.no_grad():
        model.eval()
        for batch_idx, (inputs, targets, data_indices) in enumerate(test_iterator):
            inputs = torch.tensor(inputs).cuda()
            targets = torch.tensor(targets).cuda()

            inputs = inputs.transpose(1,0)
            output = model(inputs)            
#            loss = criterion(output, targets) # not practical to calculate the loss at perframe (1-1) evaluation case
            loss = torch.zeros(1).cuda()
            
            batch_preds = []
            for j in range(output.size(0)): # batch size
                res = np.argmax(output[j].detach().cpu().numpy())
                label = targets[j].cpu().numpy()
                for k in range(targets.size(1)): # sequence size
                    outputs.append([res, label[k]])
                    batch_preds.append("{}, P-L:{}-{}".format(data_indices[j], res, label[k]))

                    t1, t5 = other.accuracy(output.detach().cpu(), targets[:,k].detach().cpu(), topk=(1,5))
                    top1.update(t1.item(), output.size(0))
                    top5.update(t5.item(), output.size(0))
            losses.update(loss.item(), output.size(0))

            other.print_and_save('[Batch {}/{}][Top1 {:.3f}[avg:{:.3f}], Top5 {:.3f}[avg:{:.3f}]]\n\t{}'.format(
                    batch_idx, len(test_iterator), top1.val, top1.avg, top5.val, top5.avg, batch_preds), log_file)
        other.print_and_save('{} Results: Loss {:.3f}, Top1 {:.3f}, Top5 {:.3f}'.format(dataset, losses.avg, top1.avg, top5.avg), log_file)
    return top1.avg, outputs

def evaluate_model(num_input, num_layers, sequence_size, batch, dropout, epochs, case, hot, lr_base, eval_dir):
    num_hidden = 2*num_input            
    model_name = 'lstm_{}_{}_{}_{}_{}_{}_{}_sgd_{}'.format(num_input, sequence_size,
                       batch, str(dropout).split('.')[0]+str(dropout).split('.')[1],
                       epochs, case, hot, 
                       str(lr_base).split('.')[0]+str(lr_base).split('.')[1])
    output_dir = os.path.join("outputs", model_name)
    log_file = os.path.join(output_dir, "results-accuracy-validation1-1.txt")
    ckpt_path = os.path.join(output_dir, model_name+"_best.pth")        
    
    model_ft = models.LSTM(num_input, num_hidden, num_layers, 7, dropout)
    model_ft = torch.nn.DataParallel(model_ft).cuda()
    checkpoint = torch.load(ckpt_path)
    model_ft.load_state_dict(checkpoint['state_dict'])
    other.print_and_save("Model loaded to gpu", log_file)
                    
    eval_loader = dataloaders.DataLoaderLSTM(eval_dir, num_input, sequence_size, 
                                             is_training=False, evaluation=True)
    eval_iterator = torch.utils.data.DataLoader(eval_loader, batch_size=batch, 
                                                num_workers=0, pin_memory=True)
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    
    top1, outputs = validate_lstm(model_ft, ce_loss, eval_iterator, checkpoint['epoch'], eval_dir.split("/")[-2] + "_Test", log_file)
    
    video_preds = [x[0] for x in outputs]
    video_labels = [x[1] for x in outputs]

    cf, recall, precision, cls_acc, mean_cls_acc, top1_acc = analyze_preds_labels(video_preds, video_labels)
    f1_macro = f1_score(video_labels, video_preds, average='macro')
    f1_all = f1_score(video_labels, video_preds, average=None)
    
    other.print_and_save(cf, log_file)
    other.print_and_save("F1 Macro {:.3f}".format(100*f1_macro), log_file)
    other.print_and_save("F1 per class {}".format(np.round(100*f1_all,3)), log_file)
    other.print_and_save("Cls Rec {}".format(recall), log_file)
    other.print_and_save("Cls Pre {}".format(precision), log_file)
    other.print_and_save("Cls Acc {}".format(cls_acc), log_file)
    other.print_and_save("Mean Cls Acc {:.02f}%".format(mean_cls_acc), log_file)
    other.print_and_save("Dataset Acc {}".format(top1_acc), log_file)

#%% FULL SCALE EVALUATION BELOW
batch=16
dropout=0.0
epochs=150
lr_base = 0.01
num_layers = 2
threshold = 3
num_input = 20
sequence_size = 20
hot = 'oh'
dataset = 'adl20'

eval_dir_oh_l = r"_data/one_hot_7_classes/labels/one_hot_{}_labels_locations/test"
eval_dir_mh_l = r"_data/multi_hot_classification/from_labels/multi_hot_7_classes/multi_hot_{}_labels_locations_sampled/test"
eval_dir_oh_d = r"_data/one_hot_7_classes/{}/one_hot_{}_detections_locations_sampled/test"
eval_dir_mh_d = r"_data/multi_hot_classification/{}/multi_hot_7_classes/multi_hot_{}_detections_locations_sampled/test"
                
case = "d2d_0{}".format(threshold)
eval_dir = eval_dir_oh_d if hot == 'oh' else eval_dir_mh_d
eval_dir = eval_dir.format(threshold, dataset)
evaluate_model(num_input, num_layers, sequence_size, batch, dropout,
               epochs, case, hot, lr_base, eval_dir)
            
