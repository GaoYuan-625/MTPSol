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
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist
from tqdm import tqdm
warnings.simplefilter('ignore', BiopythonWarning)
import matplotlib
matplotlib.use('Agg')  

class Experiment(Exp_Basic):

    def __init__(self, args):
        super(Experiment, self).__init__(args)

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
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
        return model

    def _get_data(self, flag):
        
        if self.data_init and (flag!='test' and flag!="experiment"):
            if flag == 'train':
                return  self.train_dataset, self.train_dataloader
            elif flag == 'val':
                return  self.val_dataset, self.val_dataloader
        elif flag !='test' :
            self.train_dataset, self.train_dataloader, self.val_dataset, self.val_dataloader = data_mix_provider(self.args, flag)
            self.data_init = True
            if flag == 'train':
                return  self.train_dataset, self.train_dataloader
            elif flag == 'val':
                return  self.val_dataset, self.val_dataloader
            
        if flag=='test':
            self.val_dataset, self.val_dataloader,  self.train_dataset, self.train_dataloader = None, None, None, None
            test_dataset, test_dataloader = data_provider(self.args, flag)
        elif flag=='experiment':
            self.val_dataset, self.val_dataloader,  self.train_dataset, self.train_dataloader = None, None, None, None
            test_dataset, test_dataloader = data_provider(self.args, flag)    

        return test_dataset, test_dataloader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        return criterion
 

    def test(self, setting=None, flag="test"):
        test_data, test_loader = self._get_data(flag=flag)        

        if setting is not None:
            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            checkpoint = torch.load(best_model_path, map_location=self.device)
            if self.args.use_multi_gpu:
    
                self.model.module.load_state_dict(checkpoint, strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)

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
                predicted = (probs >= 0.5).astype(float) 

                total += labels.size(0)
                correct += (predicted == labels.cpu().numpy()).sum()
                
                all_preds.extend(predicted)
                all_labels.extend(labels.detach().cpu().numpy())
                

                all_preds_prob.extend(probs)


        if self.args.use_multi_gpu:
            dist.barrier()

            def gather_all_from_all(data):
                import pickle
                data = pickle.dumps(data)
                data_size = torch.tensor(len(data), device=self.device)
                size_list = [torch.zeros_like(data_size) for _ in range(dist.get_world_size())]
                dist.all_gather(size_list, data_size)
                max_size = max([s.item() for s in size_list])
                data_tensor = torch.zeros(max_size, dtype=torch.uint8, device=self.device)
                data_tensor[:data_size] = torch.tensor(list(data), dtype=torch.uint8, device=self.device)
                data_list = [torch.zeros_like(data_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(data_list, data_tensor)
                all_data = []
                for i, tensor in enumerate(data_list):
                    size = size_list[i]
                    data_bytes = bytes(tensor[:size].cpu().numpy().tolist())
                    data_i = pickle.loads(data_bytes)
                    all_data.extend(data_i)
                return all_data

            all_preds = gather_all_from_all(all_preds)
            all_labels = gather_all_from_all(all_labels)
            all_preds_prob = gather_all_from_all(all_preds_prob)

            total_correct_tensor = torch.tensor(correct, device=self.device)
            dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
            correct = total_correct_tensor.item()
            total_tensor = torch.tensor(total, device=self.device)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            total = total_tensor.item()

        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:

            out_path = os.path.join('./out/', setting+"-"+flag)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            with open(os.path.join(out_path, f'{flag} predictions_and_labels.pkl'), 'wb') as f:
                pickle.dump({'all_preds': all_preds, 'all_labels': all_labels}, f)


            metrics = evaluate_binary_classification(all_preds, all_labels, all_preds_prob)


            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

            with open(os.path.join(out_path, f'{flag}_metrics.json'), 'w') as f:
                json.dump(metrics, f) 

            test_accuracy = correct / total

            print(f"{flag} Accuracy: {test_accuracy:.4f}")
            cm = confusion_matrix(all_labels, all_preds)
            print(f"{flag} Confusion Matrix:")
            print(cm)


            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
            plt.xlabel(f'{flag} Predicted Label')
            plt.ylabel(f'{flag} True Label')
            plt.title(f'Confusion Matrix on {flag} Set')
            plt.savefig(f'{flag} confusion_matrix_heatmap.png')


