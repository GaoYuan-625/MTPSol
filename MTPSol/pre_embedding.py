import torch
import random
import numpy as np
from torch.utils.data import DataLoader
# from models.model import ProteinAdapter
from data_provider.PreDataset import PreDataset
from data_provider.ProteinDataset import ProteinDataset
from models.sequence_embedding import SequenceEmbedding
from models.structure_embedding import parse_PDB
from models.structure_embedding import StructureEmbedding
import warnings
import argparse
import os
from tqdm import tqdm
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

def collate_fn(batch):

    sequences, structures, labels = zip(*batch)  #
 
    max_seq_len = max([seq.size(0) for seq in sequences])
    max_struc_len = max([struc.size(0) for struc in structures])
    

    padded_sequences = torch.stack([torch.cat([seq, torch.zeros(max_seq_len - seq.size(0), seq.size(1))], dim=0) for seq in sequences])
    padded_structures = torch.stack([torch.cat([struc, torch.zeros(max_struc_len - struc.size(0), struc.size(1))], dim=0) for struc in structures])

    sequence_masks = torch.stack([torch.cat([torch.ones(seq.size(0)), torch.zeros(max_seq_len - seq.size(0))]) for seq in sequences])
    structure_masks = torch.stack([torch.cat([torch.ones(struc.size(0)), torch.zeros(max_struc_len - struc.size(0))]) for struc in structures])


    labels = torch.tensor(labels)
    
    return padded_sequences, padded_structures, sequence_masks, structure_masks, labels


if __name__ == "__main__":
    seed = 42  # random seed, also can be 3407 or 114514
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description='Binary Classification Trainer')
    parser.add_argument('--gpu', type=int, default=3, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False) 
    parser.add_argument('--structureembedding_path', type=str, default="weights/v_48_020.pt", help='structure embedding path') 
    parser.add_argument('--sequenceembedding_path', type=str, default="./ESM-1v", help='structure embedding path')    
    args = parser.parse_args()


    flags = ["train", "test", "experiment"]

    for flag in flags:

        dataset_path = f"./data/{flag}"
        structureEmbedding = StructureEmbedding(args.structureembedding_path, args)

        model_locations = [f"{args.sequenceembedding_path}/esm1v_t33_650M_UR90S_1.pt",] 

        seqembedding = SequenceEmbedding(model_locations, args)
        dataset = PreDataset(dataset_path=dataset_path)

        data_loader = DataLoader(
                                    dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=1,
                                    drop_last=False,
                                    pin_memory=True,
                                )

        structure_embeddings = []
        sequence_embeddings = [] 

        for index, (batch) in tqdm(enumerate(data_loader, 0)):
            sequences, structures, labels = batch
            structure_embedding = structureEmbedding(structures)
            sequence_embedding = seqembedding(sequences)

            structure_embeddings.append(structure_embedding.detach().cpu())
            sequence_embeddings.append(sequence_embedding.detach().cpu())


        print(len(structure_embeddings))
        print(len(sequence_embeddings))
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        
        import pickle

        with open(os.path.join("./cache", f"structure_{flag}.bin"),'wb') as fp:
            pickle.dump(structure_embeddings,fp)
        
        with open(os.path.join("./cache", f"sequence_{flag}.bin"),'wb') as fp:
            pickle.dump(sequence_embeddings,fp)
        


