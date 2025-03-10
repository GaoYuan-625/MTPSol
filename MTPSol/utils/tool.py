import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch.distributed as dist
from distutils.util import strtobool
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

plt.switch_backend('agg')
def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** (epoch - 1))}    
    elif args.lradj == 'type2':
        lr_adjust = {epoch: args.learning_rate * (0.6 ** epoch)}  
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj == 'type_fix':
        if epoch < 4:
            lr_adjust = {epoch: args.learning_rate}
        elif epoch >= 4 and epoch < 6:
            lr_adjust = {epoch: args.learning_rate * 0.75 } 
        elif epoch >= 8 and epoch < 10:
            lr_adjust = {epoch: args.learning_rate * 0.5 } 
        elif epoch >= 10 and epoch < 12:
            lr_adjust = {epoch: args.learning_rate * 0.1 } 
        elif epoch >= 12 and epoch < 16:
            lr_adjust = {epoch: args.learning_rate * 0.05} 
        elif epoch >= 16 and epoch < 20:
            lr_adjust = {epoch: args.learning_rate * 0.01} 
        elif epoch >= 20 and epoch < 24:
            lr_adjust = {epoch: 5e-5} 
        elif epoch >= 24 and epoch < 30:
            lr_adjust = {epoch: 5e-6}  
        else:
            lr_adjust = {epoch: 5e-7} 
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
            print('next learning rate is {}'.format(lr))

class EarlyStopping:
    def __init__(self, args, verbose=False, delta=0):
        self.patience = args.patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.use_multi_gpu = args.use_multi_gpu
        if self.use_multi_gpu:
            self.local_rank = args.local_rank
        else:
            self.local_rank = None

    def __call__(self, val_loss, model, path):
        score = - val_loss
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            if self.use_multi_gpu:
                if self.local_rank == 0:
                    self.save_checkpoint(val_loss, model, path)
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:

            self.best_score = score
            if self.use_multi_gpu:
                if self.local_rank == 0:
                    self.save_checkpoint(val_loss, model, path)
                    print(f'Model Save: {val_loss}')
                dist.barrier()
            else:
                self.save_checkpoint(val_loss, model, path)
                print(f'Model Save: {val_loss}')
            if self.verbose:
                if (self.use_multi_gpu and self.local_rank == 0) or not self.use_multi_gpu:
                    print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            self.val_loss_min = val_loss
            self.counter = 0


    def save_checkpoint(self, val_loss, model, path):
        param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
        }
        state_dict = model.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        torch.save(state_dict, path + '/' + f'checkpoint.pth')

def evaluate_binary_classification(all_preds, all_labels, all_preds_prob=None):

    accuracy = accuracy_score(all_labels, all_preds)
    
    precision = precision_score(all_labels, all_preds)
    
    recall = recall_score(all_labels, all_preds)
    
    f1 = f1_score(all_labels, all_preds)
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    if all_preds_prob is not None:
        auc = roc_auc_score(all_labels, all_preds_prob)
        metrics['AUC'] = auc
    
    return metrics