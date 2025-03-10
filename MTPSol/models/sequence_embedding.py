import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import esm
import numpy as np
import random
import os
import sys
from typing import Any

python_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, python_path)


class SequenceEmbedding(nn.Module):
    def __init__(self, model_locations: list, args: Any):
        super(SequenceEmbedding, self).__init__()
        if args.use_multi_gpu:
            self.device = f"cuda:{args.local_rank}"
        else:
            self.device = f"cuda:{args.gpu}"

        self.models = []
        self.alphabets = []
        for model_location in model_locations:
            try:
                model, alphabet = esm.pretrained.load_model_and_alphabet(model_location)
                print(f"Loaded model {model_location}")
                model.eval()  # disables dropout for deterministic results
                if torch.cuda.is_available():
                    model = model.to(self.device)
                    print(f"Transferred model to {self.device}")

                self.models.append(model)
                self.alphabets.append(alphabet)
            except Exception as e:
                print(f"Failed to load model {model_location}")
                exit(1)
        for md in self.models:
            for name, param in md.named_parameters():
                param.requires_grad = False

    def forward(self, sequences: list):
        MAX_SEQ_LEN = 1022
        truncated_sequences = []
        for i, seq in enumerate(sequences):
            if len(seq) > MAX_SEQ_LEN:
                truncated_seq = seq[:MAX_SEQ_LEN]

                truncated_sequences.append((f"seq{i}", truncated_seq))
            else:
                truncated_sequences.append((f"seq{i}", seq))
        
        sequence_representations = []
        for model, alphabet in zip(self.models, self.alphabets):
            batch_converter = alphabet.get_batch_converter()

            batch_labels, batch_strs, batch_tokens = batch_converter(truncated_sequences)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)


            with torch.no_grad():
                results = model(batch_tokens.to(self.device), repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]


            sequence_representations.append(
                token_representations[:, 1 : token_representations.size(1) - 1, :].unsqueeze(0)
            )
        
        sequence_representations = torch.cat(sequence_representations, dim=0)

        sequence_representations = sequence_representations.mean(axis=0)

        return sequence_representations

