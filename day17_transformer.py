import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    def forward(self, x, mask = None):
        batch_size, seq_len, _ = x.shape
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked.fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim = -1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8
    x = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)
    pe = PositionalEncoding(d_model)
    x_with_pos = pe(x)
    out = mha(x_with_pos)
    print(f"输入形状：{x.shape}")
    print(f"加位置编码后：{x_with_pos.shape}")
    print(f"Multi-Head Attention 输出：{out.shape}")
    print(f"每个头的维度 d_k = {d_model} / {num_heads} = {d_model // num_heads}")
