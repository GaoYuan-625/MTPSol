import torch
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn

class CrossModalFusion(nn.Module):
    def __init__(self, query_dim, kv_dim, d_model, num_heads):
        super(CrossModalFusion, self).__init__()

        self.linear_Q = nn.Linear(query_dim, d_model)  
        self.linear_KV = nn.Linear(kv_dim, d_model) 

        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)

        self._initialize_weights()
    def forward(self, query, kv, mask=None):

        query_alin = query_proj = self.linear_Q(query)  
        kv_proj = self.linear_KV(kv) 

        attn_output, attn_output_weights = self.multihead_attn(query_proj, kv_proj, kv_proj, key_padding_mask=mask)
        

        return attn_output, query_alin, attn_output_weights

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):

                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_dim
        self.heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)
        self._initialize_weights()
    def forward(self, query, values, keys, mask=None):
        # 10 * 6  * 512
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim) 
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim) 

        values = self.values(values)       
        keys = self.keys(keys)   
        queries = self.queries(queries) 

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # import pdb;pdb.set_trace()

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class StructuralBranch(nn.Module):
    def __init__(self, structure_dim=512, sequence_dim=512, latent_dim=512):
        super(StructuralBranch, self).__init__()

        self.first_attn = CrossModalFusion(query_dim=structure_dim, kv_dim=sequence_dim, d_model=latent_dim, num_heads=4)

        self.intraLevelSelfAttn = MultiHeadAttention(embed_dim=latent_dim, num_heads=4)
        
        self.interLevelCrossAttn = MultiHeadAttention(embed_dim=latent_dim, num_heads=4)

    def forward(self, structure_emb, sequence_emb, sequence_mask=None, structure_mask=None):

        X_inter, structure_alin, _= self.first_attn(query=structure_emb, kv=sequence_emb, mask=sequence_mask)

        # print(X_inter.shape,sequence_mask.shape, structure_mask.shape)
        
        X_intra = self.intraLevelSelfAttn(X_inter, X_inter, X_inter, structure_mask.unsqueeze(1).unsqueeze(1))
        
        E_str = self.interLevelCrossAttn(X_intra, structure_alin, structure_alin, structure_mask.unsqueeze(1).unsqueeze(1))

        return E_str, structure_alin


class SequenceBranch(nn.Module):
    def __init__(self, structure_dim=512, sequence_dim=512, latent_dim=512):
        super(SequenceBranch, self).__init__()


        self.first_attn = CrossModalFusion(query_dim=sequence_dim, kv_dim=structure_dim, d_model=latent_dim, num_heads=4)

        self.intraLevelSelfAttn = MultiHeadAttention(embed_dim=latent_dim, num_heads=4)
        
        self.interLevelCrossAttn = MultiHeadAttention(embed_dim=latent_dim, num_heads=4)

    def forward(self, structure_emb, sequence_emb, sequence_mask=None, structure_mask=None):
        
        X_inter, sequence_alin, _ = self.first_attn(query=sequence_emb, kv=structure_emb, mask=structure_mask)

        X_intra = self.intraLevelSelfAttn(X_inter, X_inter, X_inter, sequence_mask.unsqueeze(1).unsqueeze(1))
        E_seq = self.interLevelCrossAttn(X_intra, sequence_alin, sequence_alin, sequence_mask.unsqueeze(1).unsqueeze(1))
        return E_seq, sequence_alin
