import torch
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn
from models.ModalFusion import StructuralBranch, SequenceBranch
from models.sequence_embedding import SequenceEmbedding
from models.structure_embedding import StructureEmbedding
from models.Pyramid import MultiLevelNetwork
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        if args.use_multi_gpu:
            self.device = f"cuda:{args.local_rank}"
        else:
            self.device = f"cuda:{args.gpu}"


        self.struct_branch = StructuralBranch(structure_dim=128, sequence_dim=1280)
        self.seq_branch = SequenceBranch(structure_dim=128, sequence_dim=1280)

 

        self.conv1 = nn.Conv1d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1024, 512, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(1024, 512, kernel_size=7, padding=3)

        self.conv_1x1 = nn.Conv1d(512 * 3, 512, kernel_size=1)

        self.bn = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.5)

        self.multilevelBlock = MultiLevelNetwork(in_channel=512, out_channels=[512, 1024, 1024], final_channel=64)

        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 1)
        self._initialize_weights()

    def forward(self, sequence, structure, sequence_mask=None, structure_mask=None):
        # print("structure shape: ", structure.shape, "sequence shape: ", sequence.shape)
        E_str, structure_alin  = self.struct_branch(structure, sequence, sequence_mask, structure_mask)  # B, N, 512
        E_seq, sequence_alin = self.seq_branch(structure, sequence, sequence_mask, structure_mask)     # B, N, 512
        
        E_str = E_str + structure_alin
        E_seq = E_seq + sequence_alin

        target_size = max(E_str.size(1), E_seq.size(1))

        E_str = F.pad(E_str, (0, 0, 0, target_size - E_str.size(1)))
        E_seq = F.pad(E_seq, (0, 0, 0, target_size - E_seq.size(1)))
        # concatenated_embeddings = torch.cat([E_str, E_seq], dim=2).permute(0, 2, 1)

        attention_weights = torch.softmax(torch.cat([E_str, E_seq], dim=-1), dim=-1)
        fused_embedding = attention_weights * torch.cat([E_str, E_seq], dim=-1)
        concatenated_embeddings = fused_embedding.permute(0, 2, 1)

        # print(concatenated_embeddings.shape)
        # exit()

        multi_scale_emb = torch.cat([
            self.conv1(concatenated_embeddings),
            self.conv2(concatenated_embeddings),
            self.conv3(concatenated_embeddings)
        ], dim=1)

        multi_level_emb = self.conv_1x1(multi_scale_emb).permute(0, 2, 1)

        # Batch Normalization 和 Dropout
        multi_level_emb = self.bn(multi_level_emb.permute(0, 2, 1)).permute(0, 2, 1)
        multi_level_emb = self.dropout(multi_level_emb)


        output = self.multilevelBlock(multi_level_emb)  # B, N + N/2 + N/4 + N/8, final_channel


        output = self.global_max_pool(output.permute(0, 2, 1)).squeeze(2)
        output = self.fc(output)  

        return output

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
