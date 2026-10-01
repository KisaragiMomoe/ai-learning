import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

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
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim = -1)
        out = attn @ v
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

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask = None):
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward = 256, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x, mask = None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerPredictor(nn.Module):
    def __init__(self, d_model = 32, num_heads = 4, num_layers = 2, seq_len = 20):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads)
        self.output_proj = nn.Linear(d_model, 1)
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.output_proj(x)

np.random.seed(42)
t = np.linspace(0, 100, 1000)
data = np.sin(t) + np.random.randn(1000) * 0.1
seq_len = 20
x = []
y = []
for i in range(len(data) - seq_len):
    x.append(data[i:i+seq_len])
    y.append(data[i+seq_len])

x = np.array(x, dtype=np.float32).reshape(-1, seq_len, 1)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

x_train_t = torch.tensor(x_train)
y_train_t = torch.tensor(y_train)
x_test_t = torch.tensor(x_test)
y_test_t = torch.tensor(y_test)

print(f"训练集：{x_train.shape} | 测试集：{x_test.shape}")

model = TransformerPredictor(d_model = 32, num_heads = 4, num_layers = 2, seq_len = seq_len)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-4)
mask = torch.tril(torch.ones(seq_len, seq_len))
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    y_pred = model(x_train_t)
    loss = loss_fn(y_pred, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test_t)
        test_loss = loss_fn(y_test_pred, y_test_t)
        test_losses.append(test_loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="训练 Loss", color="blue")
plt.plot(test_losses, label="测试 Loss", color="red")
plt.title("Loss 曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
model.eval()
with torch.no_grad():
    y_pred_all = model(torch.tensor(x, dtype=torch.float32)).numpy().flatten()

plt.plot(y.flatten(), label="真实值", color="black", alpha=0.6)
plt.plot(y_pred_all, label="预测值", color="red", alpha=0.6)
plt.title("正弦波预测对比（Transformer）")
plt.xlabel("时间步")
plt.ylabel("值")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("day19_transformer.png")
plt.show()

print("\n图片已保存：day19_transformer.png")