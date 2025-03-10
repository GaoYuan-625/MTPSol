import torch
import random
import numpy as np
from torch.utils.data import DataLoader
# from models.model import ProteinAdapter
from data_provider.ProteinDataset import ProteinDataset
from models.sequence_embedding import SequenceEmbedding
from models.structure_embedding import parse_PDB
from models.structure_embedding import StructureEmbedding
import warnings
import argparse
from data_provider.PreDataset import PreDataset
from Bio import BiopythonWarning
import os
import json

warnings.simplefilter('ignore', BiopythonWarning)

def read_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def remove_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"File {file_path} removed.")
        except Exception as e:
            print(f"removed {file_path} error: {e}")
    else:
        print(f"removed {file_path} error.")

def clean_failed_data(json_file_path):
    data = read_json(json_file_path)
    
    for item in data:
        path = item[0]
        if os.path.exists(path):
            print(path)
            remove_file(path)
        else:
            print(f"Path {path} error.")


if __name__ == "__main__":
    seed = 42  # random seed, also can be 3407 or 114514
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description='Binary Classification Trainer')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)   
    args = parser.parse_args()
    
    checkpoint_path = "./weights/v_48_020.pt"

    structureEmbedding = StructureEmbedding(checkpoint_path, args)


    flags = ["train", "test", "experiment"]

    for flag in flags:

        # Create a dataset from a directory of files
        dataset_path = f"./data/{flag}"

        dataset = PreDataset(dataset_path=dataset_path)

        # Create a DataLoader to handle batching of the dataset
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )


        failed = []
        from tqdm import tqdm 
        for index, (batch) in tqdm(enumerate(data_loader, 0)):
            sequences, structures, labels = batch
            try:
                out = structureEmbedding(structures)
            except Exception as e:
                print(structures)
                failed.append(structures)
            import json

            with open(f"data_{flag}_failed.json", "w") as f:
                json.dump(failed, f, ensure_ascii=False, indent=4)



        json_file_path = f'data_{flag}_failed.json'  

        clean_failed_data(json_file_path)