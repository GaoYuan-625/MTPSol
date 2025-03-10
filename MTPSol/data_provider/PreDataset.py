import os
import sys
import pandas as pd


import torch

# torch.multiprocessing.set_start_method("spawn")

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


class PreDataset(Dataset):
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

        # lable_path = os.path.join(dataset_path, "set.csv")

        # self.label_df = pd.read_csv(lable_path)
        lable_path = os.path.join(dataset_path, "set.xlsx")

        self.label_df = pd.read_excel(lable_path)
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
        structure = self.file_paths[idx]
        sequence = self.extract_sequence(structure)
        
        # label = \
        # self.label_df.loc[self.label_df["sid"] == int(os.path.split(structure)[-1].split(".")[0]), "solubility"].iloc[0]
        label = \
        self.label_df.loc[self.label_df["sid"] == str(os.path.split(structure)[-1].split(".")[0]), "solubility"].iloc[0]
        return sequence, structure, label



