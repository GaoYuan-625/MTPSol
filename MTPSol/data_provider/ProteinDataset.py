


import os
import sys
import pandas as pd

# python_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# sys.path.insert(0, python_path)

import torch

# torch.multiprocessing.set_start_method("spawn")
import pickle
from Bio.PDB import PDBParser
from torch.utils.data import Dataset, DataLoader
from models.sequence_embedding import SequenceEmbedding
import json
# from models.structure_embedding import StructureEmbedding

# Mapping from three-letter residue names to one-letter codes
residue_mapping = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


class ProteinDataset(Dataset):
    def __init__(self, dataset_path):

        if os.path.exists(os.path.join(dataset_path, "pdb.json")):
            with open(os.path.join(dataset_path, "pdb.json"), "r") as f:
                self.file_paths = json.load(f)
        else: 
            self.file_paths = [os.path.join(dataset_path, "Pdb", f)
                for f in os.listdir(os.path.join(dataset_path, "Pdb"))
                if os.path.isfile(os.path.join(dataset_path, "Pdb", f))
            ]
            with open(os.path.join(dataset_path, "pdb.json"), "w") as f:
                json.dump(self.file_paths, f)

        if os.path.exists(os.path.join(dataset_path, "set.csv")):
            lable_path = os.path.join(dataset_path, "set.csv")
            self.label_df = pd.read_csv(lable_path)
        else:
            lable_path = os.path.join(dataset_path, "set.xlsx")
            self.label_df = pd.read_excel(lable_path)

        if "train" in dataset_path:
            structure_path = "./cache/structure_train.bin"
            sequence_path = "./cache/sequence_train.bin"
        elif "test" in dataset_path:
            structure_path = "./cache/structure_test.bin"
            sequence_path = "./cache/sequence_test.bin"
        elif "experiment" in dataset_path:
            structure_path = "./cache/structure_experiment.bin"
            sequence_path = "./cache/sequence_experiment.bin"
        else:
            raise ValueError("UNKNOWN DATASET PATH")
        with open(structure_path,'rb') as fp:
            self.structure_embeddings=pickle.load(fp)
            print("Structure_embeddings Len is ", len(self.structure_embeddings))

        with open(sequence_path,'rb') as sp:
            self.sequence_embeddings=pickle.load(sp)
            print("Sequence_embeddings Len is ", len(self.sequence_embeddings))

    def __len__(self):
        return len(self.file_paths)

    def extract_sequence(self, pdb_file):
        parser = PDBParser()
        # Parse the structure
        structure = parser.get_structure("protein", pdb_file)

        # Extract the sequence
        sequence = ""
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.get_resname() in residue_mapping:
                        sequence += residue_mapping[residue.get_resname()]

        return sequence

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        structure =  self.structure_embeddings[idx]
        sequence =  self.sequence_embeddings[idx]
        label = \
        self.label_df.loc[self.label_df["sid"] == int(os.path.split(file_path)[-1].split(".")[0]), "solubility"].iloc[0]

        return sequence.squeeze(0), structure.squeeze(0), label



