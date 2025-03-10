import torch
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn
from models.ModalFusion import StructuralBranch, SequenceBranch
from models.sequence_embedding import SequenceEmbedding
from models.structure_embedding import StructureEmbedding
from torch.nn import functional as F
# Tag Drop MultiLevelNetwork
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        
        if args.use_multi_gpu:
            self.device = f"cuda:{args.local_rank}"
        else:
            self.device = f"cuda:{args.gpu}"

        self.struct_branch = StructuralBranch(structure_dim=128, sequence_dim=1280)
        self.seq_branch = SequenceBranch(structure_dim=128, sequence_dim=1280)

        self.map = nn.Linear(1024, 64)

        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 1)
        self._initialize_weights()

    def forward(self, sequence, structure, sequence_mask=None, structure_mask=None):

        E_str, structure_alin  = self.struct_branch(structure, sequence, sequence_mask, structure_mask)  # B, N, 512
        E_seq, sequence_alin = self.seq_branch(structure, sequence, sequence_mask, structure_mask)     # B, N, 512
        
        E_str = E_str + structure_alin
        E_seq = E_seq + sequence_alin


        target_size = max(E_str.size(1), E_seq.size(1))

        E_str = F.pad(E_str, (0, 0, 0, target_size - E_str.size(1)))
        E_seq = F.pad(E_seq, (0, 0, 0, target_size - E_seq.size(1)))


        attention_weights = torch.softmax(torch.cat([E_str, E_seq], dim=-1), dim=-1)
        fused_embedding = attention_weights * torch.cat([E_str, E_seq], dim=-1)
        output = self.map(fused_embedding)

        output = self.global_max_pool(output.permute(0, 2, 1)).squeeze(2)
        output = self.fc(output)  

        return output

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
