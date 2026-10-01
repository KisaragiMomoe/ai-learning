import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

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

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout = 0.3):
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

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model = 64, num_heads = 4, num_layers = 3, max_len = 128, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward = 4 * d_model, dropout = dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len).to(x.device))
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

text = """
人工智能是研究如何让计算机完成需要人类智能的任务的学科。
它起源于二十世纪五十年代。早期的人工智能主要基于规则。
后来出现了机器学习。机器学习让计算机从数据中学习规律。
深度学习是机器学习的一个分支。它使用多层神经网络。
神经网络由很多神经元组成。每个神经元接收输入，进行计算，然后输出。
深度学习在图像识别、语音识别、自然语言处理等领域取得了巨大成功。
近年来，大语言模型成为人工智能的热点。大语言模型用海量文本训练。
它们能生成流畅的文本。它们能回答问题，写文章，甚至写代码。
但是大语言模型也有缺点。它们可能产生错误信息。它们需要大量计算资源。
人工智能正在快速发展。未来会有更多应用。
人工智能正在改变我们的世界。从智能手机到自动驾驶汽车，从医疗诊断到金融分析，人工智能无处不在。
人工智能的核心是让计算机模拟人类的智能行为。这些行为包括学习、推理、感知和决策。
机器学习是实现人工智能的重要方法。它让计算机从数据中自动学习规律。
深度学习使用多层神经网络来提取数据的特征。
通过反向传播算法，神经网络可以自动调整权重。
卷积神经网络可以识别图片中的物体。
循环神经网络和Transformer模型被广泛使用。
Transformer模型使用注意力机制。注意力机制让模型关注输入序列中的重要部分。
大语言模型是基于Transformer的模型。它们用海量文本数据训练。
大语言模型可以生成文本、回答问题、翻译语言、写代码。
但是大语言模型也有局限性。它们可能生成错误的信息。
训练一个大语言模型需要数千个GPU。
人工智能的发展也带来了伦理问题。我们需要确保人工智能被安全、公平地使用。
未来，人工智能将继续发展。它将在更多领域发挥作用。
我们需要学习人工智能的知识，以便更好地利用它。
"""

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print(f"文本长度：{len(text)}")
print(f"字符表大小：{vocab_size}")
print(f"字符表：{''.join(chars)}")

data = torch.tensor([char_to_idx[ch] for ch in text], dtype = torch.long)
seq_len = 32
x = []
y = []
for i in range(len(data) - seq_len):
    x.append(data[i : i + seq_len])
    y.append(data[i + 1 : i + seq_len + 1])
x = torch.stack(x)
y = torch.stack(y)

print(f"训练样本数：{x.shape[0]}")
print(f"每个样本长度：{x.shape[1]}")

split = int(len(x) * 0.9)
x_train, x_val = x[:split], x[split:]
y_train, y_val = y[:split], y[split:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
model = MiniGPT(vocab_size, d_model = 64, num_heads = 4, num_layers = 3, max_len = seq_len).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001, weight_decay = 0.05)
epochs = 200
batch_size = 32
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    indices = torch.randint(0, len(x_train), (batch_size,))
    x_batch = x_train[indices].to(device)
    y_batch = y_train[indices].to(device)
    logits = model(x_batch)
    loss = loss_fn(logits.view(-1, vocab_size), y_batch.view(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        val_indices = torch.randint(0, len(x_val), (batch_size,))
        x_val_batch = x_val[val_indices].to(device)
        y_val_batch = y_val[val_indices].to(device)
        val_logits = model(x_val_batch)
        val_loss = loss_fn(val_logits.view(-1, vocab_size), y_val_batch.view(-1))
        val_losses.append(val_loss.item())
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

def generate(model, start_text, max_new_token = 50, temperature = 1.0):
    model.eval()
    input_ids = torch.tensor([char_to_idx[ch] for ch in start_text], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new_token):
            input_crop = input_ids[:, -seq_len:]
            logits = model(input_crop)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim = -1)
            next_id = torch.multinomial(probs, num_samples = 1)
            input_ids = torch.cat([input_ids, next_id], dim = 1)
    generated = "".join([idx_to_char[i.item()] for i in input_ids[0]])
    return generated

print("\n" + "="*50)
print("生成示例：")
print("="*50)
for start in ["人工智能", "机器学习", "大语言模型", "未来"]:
    print(f"\n起始：{start}")
    print(generate(model, start, max_new_token = 30, temperature = 0.7))

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="训练 Loss", color="blue", alpha=0.7)
plt.plot(val_losses, label="验证 Loss", color="red", alpha=0.7)
plt.title("Mini GPT 训练 Loss 曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("day20_mini_gpt.png")
plt.show()

print("\n图片已保存：day20_mini_gpt.png")