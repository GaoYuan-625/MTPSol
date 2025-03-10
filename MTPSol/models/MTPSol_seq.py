import torch
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn
from models.ModalFusion import StructuralBranch, SequenceBranch
from models.Pyramid import MultiLevelNetwork
from torch.nn import functional as F
# Tag Drop StructuralBranch
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        if args.use_multi_gpu:
            self.device = f"cuda:{args.local_rank}"
        else:
            self.device = f"cuda:{args.gpu}"

        self.seq_branch = SequenceBranch(structure_dim=128, sequence_dim=1280)

        self.conv1 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=7, padding=3)

        self.conv_1x1 = nn.Conv1d(512 * 3, 512, kernel_size=1)

        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)


        self.multilevelBlock = MultiLevelNetwork(in_channel=512, out_channels=[512, 1024, 1024], final_channel=64)

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 1)
        self._initialize_weights()

    def forward(self, sequence, structure, sequence_mask=None, structure_mask=None):


        E_seq, seq_alin  = self.seq_branch(structure, sequence, sequence_mask, structure_mask)  # B, N, 512
        E_seq = E_seq + seq_alin

        attention_weights = torch.softmax(E_seq, dim=-1)
        fused_embedding = attention_weights * E_seq

        concatenated_embeddings = fused_embedding.permute(0, 2, 1)
        
        multi_scale_emb = torch.cat([
            self.conv1(concatenated_embeddings),
            self.conv2(concatenated_embeddings),
            self.conv3(concatenated_embeddings)
        ], dim=1)

        multi_level_emb = self.conv_1x1(multi_scale_emb).permute(0, 2, 1)

        multi_level_emb = self.bn(multi_level_emb.permute(0, 2, 1)).permute(0, 2, 1)
        multi_level_emb = self.dropout(multi_level_emb)


        output = self.multilevelBlock(multi_level_emb)  


        output = self.global_max_pool(output.permute(0, 2, 1)).squeeze(2)
        output = self.fc(output)  

        return output

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
