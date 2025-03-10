import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size=3, stride=2, padding=1):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class LinearAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads: int = 3, latent_dim: int = 512, dropout=0.1):
        super(LinearAttention, self).__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        # Linear projections
        self.W_q = nn.Linear(query_dim, latent_dim * num_heads, bias=False)
        self.W_k = nn.Linear(key_dim, latent_dim * num_heads, bias=False)
        self.W_v = nn.Linear(value_dim, latent_dim * num_heads, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(latent_dim * num_heads, latent_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Linear projections
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.latent_dim)
        K = self.W_k(key).view(batch_size, seq_len, self.num_heads, self.latent_dim)
        V = self.W_v(value).view(batch_size, seq_len, self.num_heads, self.latent_dim)

        # Transpose for dot product: (batch_size, num_heads, seq_len, latent_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.latent_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Compute the output: (batch_size, num_heads, seq_len, latent_dim)
        attention_output = torch.matmul(attention_probs, V)

        # Concatenate heads and apply dropout
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, -1)
        attention_output = self.fc(attention_output)
        attention_output = self.dropout(attention_output)

        return attention_output


class TransformerEncodeLayer(nn.Module):
    def __init__(self, d_model, latent_dim, num_heads=3, downsample_factor=4, forward_expansion=4, max_len=10000):
        super(TransformerEncodeLayer, self).__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.avg_pool = nn.AvgPool1d(kernel_size=downsample_factor, stride=downsample_factor)
        self.transformer_encoder = LinearAttention(query_dim=d_model, key_dim=d_model // 4, value_dim=d_model // 4,
                                                   num_heads=num_heads, latent_dim=latent_dim)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(latent_dim, forward_expansion * latent_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * latent_dim, latent_dim)
        )

    def forward(self, x):
        x = x + self.pos_encoder[:, :x.size(1), :]
        normed_x = self.layernorm1(x)
        pooled_x = self.avg_pool(normed_x)
        attentioned_x = self.transformer_encoder(normed_x, pooled_x, pooled_x)
        x = x + attentioned_x
        x = self.layernorm2(x + self.feed_forward(x))
        return x


class MultiLevelBlock(nn.Module):
    def __init__(self, in_channel, out_channel, latent_dim):
        super(MultiLevelBlock, self).__init__()
        self.patch_emb = PatchEmbedding(in_channel=in_channel, out_channel=out_channel)
        self.transformer_encoder = TransformerEncodeLayer(d_model=out_channel, latent_dim=latent_dim)

    def forward(self, x):
        x = self.patch_emb(x)
        x = self.transformer_encoder(x)
        return x


class MultiLevelNetwork(nn.Module):
    def __init__(self, in_channel: int, out_channels: list[int], final_channel: int):
        super(MultiLevelNetwork, self).__init__()

        multiLevelLayers = []

        linearLayers = [nn.Linear(in_channel, final_channel)]

        last_layer_size = in_channel

        for out_channel in out_channels:
            multiLevelLayers.append(
                MultiLevelBlock(in_channel=last_layer_size, out_channel=out_channel, latent_dim=out_channel)
            )
            linearLayers.append(nn.Linear(out_channel, final_channel))
            last_layer_size = out_channel

        self.multiLevelLayers = nn.ModuleList(multiLevelLayers)
        self.linearLayers = nn.ModuleList(linearLayers)

    def forward(self, x):
        layerLinearOutput = [self.linearLayers[0](x)]

        for idx, layer in enumerate(self.multiLevelLayers):
            x = layer(x)
            layerLinearOutput.append(self.linearLayers[idx + 1](x))

        final_output = torch.cat(layerLinearOutput, dim=1)

        return final_output
