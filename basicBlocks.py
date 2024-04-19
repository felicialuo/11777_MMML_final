import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):

        super().__init__()
        self.embed_dim = embed_dim
        # This class assumes that the input dimension for query, key and value is embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape

        # project query, key and value
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # compute dot-product attention + scaling value
        # Expected shape of dot_product is (N, S, T)
        dot_product = torch.einsum('nsd,ndt->nst', query, key.permute(0, 2, 1)) # (N, S, D) @ (N, D, T)
        dot_product =  dot_product / (self.embed_dim ** 0.5)

        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            additive_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            dot_product += additive_mask

        # apply softmax, dropout, and use value
        dot_product = F.softmax(dot_product, dim=-1) # expected (N, S, T)
        y = torch.einsum('nst,ntd->nsd', self.dropout(dot_product), value) # (N, S, T) @ (N, T, D)

        return y

class MultiHeadAttentionLayer(AttentionLayer):

    def __init__(self, embed_dim, num_heads, dropout=0.1):

        super().__init__(embed_dim, dropout)
        self.num_heads = num_heads
        self.dim_phead = embed_dim // num_heads

        self.head_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        H = self.num_heads
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape

        # project query, key and value
        # after projection, split the embedding across num_heads
        # expected shape for value is (N, H, T, D/H)
        query = self.query_proj(query)
        query = query.reshape(N, S, self.num_heads, self.dim_phead).permute(0, 2, 1, 3)
        # exp'd (N, H, S, D/H)

        key = self.key_proj(key)
        key = key.reshape(N, T, self.num_heads, self.dim_phead).permute(0, 2, 1, 3)
        # exp'd (N, H, T, D/H)

        value = self.value_proj(value)
        value = value.reshape(N, T, self.num_heads, self.dim_phead).permute(0, 2, 1, 3)
        # exp'd (N, H, T, D/H)

        # compute dot-product attention separately for each head. Don't forget the scaling value!
        # Expected shape of dot_product is (N, H, S, T)
        dot_product = torch.einsum('nhsd,nhtd->nhst', query, key) / (self.dim_phead ** 0.5)
        # (N, H, S, D/H) @ (N, H, D/H, T)

        if attn_mask is not None:
            # convert att_mask which is multiplicative, to an additive mask
            additive_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
            dot_product += additive_mask

        # apply softmax, dropout, and use value
        dot_product = F.softmax(dot_product, dim=-1) # expected (N, H, S, T)
        y = torch.einsum('nhst,nhtd->nhsd', self.dropout(dot_product), value) # (N, H, S, T) @ (N, H, T, D/H)

        # concat embeddings from different heads, and project
        y = y.permute(0, 2, 1, 3).reshape(N, S, D)
        output = self.head_proj(y)
        return output
    
class SelfAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn =  MultiHeadAttentionLayer(input_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(input_dim)

    def forward(self, seq, mask=None):
        # with residual connection
        out = self.self_attn(seq, seq, seq, attn_mask=mask)
        out = self.dropout(out)
        out += seq
        out = self.layernorm(out)

        return out

class CrossAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, seq, cond):
        # wih residual connection
        out = self.cross_attn(seq, cond, cond)
        out = self.dropout(out)
        out += seq
        out = self.norm(out)
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=1024, dropout=0.1 ):
        super().__init__()
        # 2-layer MLP, hidden dim of linear is given by dim_feedforward
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(input_dim)


    def forward(self, seq):
        # with residual connection
        out = self.mlp(seq)
        out = self.dropout(out)
        out += seq
        out = self.norm(out)
        return out

class SelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim, num_heads, dropout)
        self.feedforward_block = FeedForwardBlock(input_dim, num_heads, dim_feedforward, dropout)

    def forward(self, seq, mask=None):
        out = self.self_atn_block(seq, mask)
        return self.feedforward_block(out)

class CrossAttentionEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim, num_heads, dropout)
        self.cross_atn_block = CrossAttentionBlock(input_dim, num_heads, dropout)
        self.feedforward_block = FeedForwardBlock(input_dim, num_heads, dim_feedforward, dropout)

    def forward(self, seq, cond, mask=None):
        out = self.cross_atn_block(seq, cond)
        out = self.self_atn_block(out, mask)
        return self.feedforward_block(out)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, len=16, dropout=0.1):
        super().__init__()
        self.encoding = nn.Embedding(len, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        positions = torch.arange(len).unsqueeze(0)
        self.register_buffer("positions", positions)

    def forward(self, x):
        B, L, D = x.shape
        pos = self.positions[:, :L]
        pos_emb = self.encoding(pos)

        out = x + pos_emb
        out = self.dropout(out)

        return out
 