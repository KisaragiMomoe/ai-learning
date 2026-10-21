import torch
import torch.nn.functional as F
import math

def flash_attention_simulation(Q, K, V, block_size = 2):
    seq_len, d_k = Q.shape
    O = torch.zeros_like(Q)
    l = torch.zeros((seq_len, 1))
    m = torch.full((seq_len, 1), -float('inf'))

    for j in range(0, seq_len, block_size):
        K_block = K[j:j + block_size, :]
        V_block = V[j:j + block_size, :]

        S_block = Q @ K_block.T / math.sqrt(d_k)

        m_new = torch.maximum(m, S_block.max(dim = -1, keepdim = True).values)

        correction = torch.exp(m - m_new)
        l = l * correction
        O = O * correction

        P_block = torch.exp(S_block - m_new)
        l = l + P_block.sum(dim = -1, keepdim = True)
        O = O + P_block @ V_block

        m = m_new

    O = O / l
    return O

seq_len = 6
d_k = 4
Q = torch.randn(seq_len, d_k)
K = torch.randn(seq_len, d_k)
V = torch.randn(seq_len, d_k)

# 标准 Attention
scores = Q @ K.T / math.sqrt(d_k)
standard_attn = F.softmax(scores, dim=-1) @ V

# 模拟 FlashAttention
flash_attn = flash_attention_simulation(Q, K, V, block_size=2)

print("标准注意力输出:\n", standard_attn)
print("\nFlash注意力输出:\n", flash_attn)
print("\n两者的最大误差:", (standard_attn - flash_attn).abs().max().item())