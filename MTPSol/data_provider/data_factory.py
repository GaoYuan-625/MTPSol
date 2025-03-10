from data_provider.ProteinDataset import ProteinDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torch

def collate_fn(batch):

    sequences, structures, labels = zip(*batch)  
    

    max_seq_len = max([seq.size(0) for seq in sequences])
    max_struc_len = max([struc.size(0) for struc in structures])

    padded_sequences = torch.stack([torch.cat([seq, torch.zeros(max_seq_len - seq.size(0), seq.size(1))], dim=0) for seq in sequences])
    padded_structures = torch.stack([torch.cat([struc, torch.zeros(max_struc_len - struc.size(0), struc.size(1))], dim=0) for struc in structures])
    

    sequence_masks = torch.stack([torch.cat([torch.ones(seq.size(0)), torch.zeros(max_seq_len - seq.size(0))]) for seq in sequences])
    structure_masks = torch.stack([torch.cat([torch.ones(struc.size(0)), torch.zeros(max_struc_len - struc.size(0))]) for struc in structures])


    labels = torch.tensor(labels)
    
    return padded_sequences, padded_structures, sequence_masks, structure_masks, labels

def data_provider(args, flag):

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size 
    elif flag == 'val':
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size 
    elif flag == 'experiment':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size 
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    data_set = ProteinDataset(dataset_path=f"./data/{flag}")

    if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
        print(flag, len(data_set))

    if args.use_multi_gpu:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(data_set, 
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=drop_last,
            )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last)
        
    return data_set, data_loader

def data_mix_provider(args, flag):
    shuffle_flag = True
    drop_last = False
    batch_size = args.batch_size

    data_set = ProteinDataset(dataset_path=f"./data/train")
    train_size = int(args.vali_rate * len(data_set))
    val_size = len(data_set) - train_size 
    train_dataset, val_dataset, = random_split(data_set, [train_size, val_size])

    if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
        print('train', len(train_dataset))
        print('val', len(val_dataset))

    if args.use_multi_gpu:
        train_datasampler = DistributedSampler(train_dataset, shuffle=shuffle_flag)
        train_loader = DataLoader(train_dataset, 
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_fn,
            )
        
        val_datasampler = DistributedSampler(val_dataset, shuffle=shuffle_flag)
        val_loader = DataLoader(val_dataset, 
            batch_size=batch_size,
            sampler=val_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
            collate_fn=collate_fn,
            )
    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last)
        
    return train_dataset, train_loader,val_dataset, val_loader