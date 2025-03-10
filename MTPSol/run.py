import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import confusion_matrix
import numpy as np
import random
import torch.distributed as dist
from exp.Exp import Experiment
from exp.Exp_Single import Experiment_Single

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

set_seed(2024)

def main():

    parser = argparse.ArgumentParser(description='Binary Classification Trainer')
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model', type=str,  default='ProteinClassification', help='status')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--print_every', type=int, default=10, help='Print loss every n batches')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    parser.add_argument('--lradj', type=str, default='type_fix', help='adjust learning rate')
    parser.add_argument('--vali_rate', type=float, default=0.9, help='validation rate')    
    parser.add_argument('--warmup_steps', type=int, default=500, help='warmup steps')    
    parser.add_argument('--structureembedding_path', type=str, default="weights/v_48_020.pt", help='structure embedding path') 
    parser.add_argument('--sequenceembedding_path', type=str, default="./ESM-1v", help='structure embedding path')    
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    parser.add_argument('--lambdalr', action='store_true', help='use lambdalr annealing lr', default=False)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)   
    parser.add_argument('--gradient_accumulation', action='store_true', help='Gradient accumulation', default=False)   
    parser.add_argument('--gradient_accumulation_step', type=int, default=4, help='Gradient accumulation step')   
    parser.add_argument('--checkpoints', type=str, default='./best-checkpoints/checkpoints.pth', help='location of model checkpoints')
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    
    config = parser.parse_args()
    if config.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")

        port = os.environ.get("MASTER_PORT", "64209")

        hosts = int(os.environ.get("WORLD_SIZE", "8"))

        rank = int(os.environ.get("RANK", "0")) 

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        gpus = torch.cuda.device_count()

        config.local_rank = local_rank

        print(ip, port, hosts, rank, local_rank, gpus)

        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)

        torch.cuda.set_device(local_rank)

    if config.is_training:
        if config.use_multi_gpu:
            exp = Experiment(config)
        else:
            print("Single GPU training")
            exp = Experiment_Single(config)
        setting = 'Name: {}_B{}_L{}_N{}_PE{}_P{}_Lr{}_Cos{}_Vrate_{}_WarmS{}'.format(
                    config.model,
                    config.batch_size,
                    config.learning_rate,
                    config.num_epochs,
                    config.print_every,
                    config.patience,
                    config.lradj,
                    config.cosine,
                    config.vali_rate,
                    config.warmup_steps,
                    )
        if (config.use_multi_gpu and config.local_rank == 0) or not config.use_multi_gpu:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        if (config.use_multi_gpu and config.local_rank == 0) or not config.use_multi_gpu:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

        exp.test(setting,flag='test')
        if (config.use_multi_gpu and config.local_rank == 0) or not config.use_multi_gpu:
            print('>>>>>>>experiment : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, flag="experiment")
        torch.cuda.empty_cache()
    else:
        setting = 'Name: {}_B{}_L{}_N{}_PE{}_P{}_Lr{}_Cos{}_Vrate_{}_WarmS{}'.format(
                    config.model,
                    config.batch_size,
                    config.learning_rate,
                    config.num_epochs,
                    config.print_every,
                    config.patience,
                    config.lradj,
                    config.cosine,
                    config.vali_rate,
                    config.warmup_steps,
                    )
        if config.use_multi_gpu:
            exp = Experiment(config)
        else:
            print("Single GPU training")
            exp = Experiment_Single(config)
            
        if (config.use_multi_gpu and config.local_rank == 0) or not config.use_multi_gpu:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))       
        exp.test(setting,flag='test')
        if (config.use_multi_gpu and config.local_rank == 0) or not config.use_multi_gpu:
            print('>>>>>>>experiment : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, flag="experiment")

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

