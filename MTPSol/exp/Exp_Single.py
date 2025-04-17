import json
import math
import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.metrics import confusion_matrix
import numpy as np
from data_provider.data_factory import data_provider, data_mix_provider
from exp.exp_basic import Exp_Basic
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from Bio import BiopythonWarning
from utils.tool import EarlyStopping, adjust_learning_rate, evaluate_binary_classification
import os
from tqdm import tqdm
warnings.simplefilter('ignore', BiopythonWarning)
import matplotlib
matplotlib.use('Agg') 


class HybridLossWithPositiveConstraints(nn.Module):
    def __init__(self, weight_init=0.5, alpha=1.0, gamma=2.0, pos_weight_init=None, neg_weight_init=None):
        """
        :param alpha: Focal Loss alpha 
        :param gamma: Focal Loss gamma 
        :param pos_weight_init
        :param neg_weight_init
        """
        super(HybridLossWithPositiveConstraints, self).__init__()
        self.weight_init = nn.Parameter(torch.tensor(weight_init, requires_grad=True))
        self.alpha = alpha
        self.gamma = gamma
        self.softplus = nn.Softplus()

        self.pos_weight = pos_weight_init if pos_weight_init is not None else torch.tensor(1.0)
        self.neg_weight = neg_weight_init if neg_weight_init is not None else torch.tensor(1.0)
    
    def forward(self, logits, labels):
        weight = self.softplus(self.weight_init)

        bce_loss_each = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        
        pos_weight = self.pos_weight.to(logits.device)
        neg_weight = self.neg_weight.to(logits.device)
        sample_weights = torch.where(labels == 1, pos_weight, neg_weight)
        
        bce_loss_weighted = bce_loss_each * sample_weights

        probs = torch.sigmoid(logits)
        focal_weight = self.alpha * (1 - probs) ** self.gamma

        focal_loss = focal_weight * bce_loss_weighted

        loss = weight * bce_loss_weighted + (1 - weight) * focal_loss

        loss += 0.05 * (self.weight_init ** 2)

        return loss.mean()


def save_last_checkpoint(model, path):
    param_grad_dic = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            # delete parameters that do not require gradient
            del state_dict[k]
    torch.save(state_dict, path + '/' + f'last_checkpoint.pth')

class Experiment_Single(Exp_Basic):

    def __init__(self, args):
        super(Experiment_Single, self).__init__(args)

        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

        plt.ioff()

        self.fig_epoch = plt.figure(num='Epoch Figure', figsize=(8, 8))
        self.axs_epoch = self.fig_epoch.subplots(2, 1)
        self.fig_batch = plt.figure(num='Batch Figure', figsize=(8, 8))
        self.axs_batch = self.fig_batch.subplots(2, 1)

        plt.show(block=False)


        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        self.batch_losses = []
        self.batch_accuracies = []
        self.batch_iters = []

        self.data_init = False

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        self.device = self.args.gpu
        model = model.to(self.device)
        return model

    def _get_data(self, flag):
        if self.data_init and (flag!='test' and flag!="experiment"):
            if flag == 'train':
                return self.train_dataset, self.train_dataloader
            elif flag == 'val':
                return self.val_dataset, self.val_dataloader

        elif flag!='test' and flag!="experiment":
            self.train_dataset, self.train_dataloader, self.val_dataset, self.val_dataloader = data_mix_provider(self.args, flag)
            self.data_init = True
            if flag == 'train':
                return  self.train_dataset, self.train_dataloader
            elif flag == 'val':
                return  self.val_dataset, self.val_dataloader
            
        if flag == 'test':
            self.val_dataset, self.val_dataloader,  self.train_dataset, self.train_dataloader = None, None, None, None
            self.data_init = False
            test_dataset, test_dataloader = data_provider(self.args, flag)
        elif flag=='experiment':
            self.val_dataset, self.val_dataloader,  self.train_dataset, self.train_dataloader = None, None, None, None
            self.data_init = False
            test_dataset, test_dataloader = data_provider(self.args, flag)    

        return test_dataset, test_dataloader

    def _select_optimizer(self, criterion=None):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append({'params': p, 'name': n}) 
                print(n, p.dtype, p.shape)
        
        if criterion:
            if hasattr(criterion, 'raw_alpha'):
                p_list.append({'params': [criterion.raw_alpha], 'name': 'raw_alpha', 'weight_decay': 0.0})
                print("Added raw_alpha to optimizer with shape:", criterion.raw_alpha.shape)

        model_optim = optim.AdamW(p_list, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        print('Next learning rate is {}'.format(self.args.learning_rate))
        return model_optim
    
    def _select_criterion(self):
        pos_weight = torch.tensor([1.0]).to(self.device)  
        neg_weight = torch.tensor([2.0]).to(self.device)  
        criterion = HybridLossWithPositiveConstraints(
            weight_init=0.5,
            alpha=0.75,
            gamma=2.5,
            pos_weight_init=pos_weight,
            neg_weight_init=neg_weight
        )
        return criterion

   
    def test(self, setting=None, flag="test"):
        test_data, test_loader = self._get_data(flag=flag)    
        self.out_path = os.path.join('./out/', setting)

        if setting is not None:

            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            # best_model_path = os.path.join('./best-checkpoints', 'checkpoint.pth')
            # best_model_path = os.path.join(self.args.checkpoints, setting, 'last_checkpoint.pth')
            # to adpater to different dataset
            threshold = 0.25 if flag =='experiment' else 0.40
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
            print(f"Model **********{best_model_path} ***********loaded")
        
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_preds_prob = []

        with torch.no_grad():
            for padded_sequences, padded_structures, sequence_masks, structure_masks, labels in test_loader:
                padded_sequences = padded_sequences.to(self.device)
                padded_structures = padded_structures.to(self.device)
                sequence_masks = sequence_masks.to(self.device)
                structure_masks = structure_masks.to(self.device)

                labels = labels.to(self.device).view(-1).float()
                outputs = self.model(padded_sequences, padded_structures, sequence_masks, structure_masks).view(-1)

                probs = torch.sigmoid(outputs).detach().cpu().numpy()

                predicted = (probs >= threshold).astype(float)

                total += labels.size(0)
                correct += (predicted == labels.cpu().numpy()).sum()

                all_preds.extend(predicted)
                all_labels.extend(labels.detach().cpu().numpy())

                all_preds_prob.extend(probs)
                
        out_path = os.path.join('./out/', setting)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        with open(os.path.join(out_path, f'{flag} predictions_and_labels.pkl'), 'wb') as f:
            pickle.dump({'all_preds': all_preds, 'all_labels': all_labels}, f)

        metrics = evaluate_binary_classification(all_preds, all_labels, all_preds_prob)

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        with open(os.path.join(out_path,  f'{flag} metrics.json'), 'w') as f:
            json.dump(metrics, f)

        test_accuracy = correct / total
        print(f"{flag} Accuracy: {test_accuracy:.4f}")
        cm = confusion_matrix(all_labels, all_preds)
        print(f"{flag} Confusion Matrix:")
        print(cm)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Insoluble', 'Soluble'], yticklabels=['Insoluble', 'Soluble'])
        plt.xlabel(f'{flag} Predicted Label')
        plt.ylabel(f'{flag} True Label')
        plt.title(f'Confusion Matrix on {flag} Set')

        plt.savefig(os.path.join(self.out_path, f'{flag} confusion_matrix_heatmap.png'))
