"""
Day 36: 混合精度训练
对比 fp32 和 AMP 的速度、显存。
"""

import torch
import torch.nn as nn
import time

# ========== 1. 检查设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备：{device}")

if device.type == "cuda":
    print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"支持 bf16：{torch.cuda.is_bf16_supported()}")
else:
    print("没有 GPU，混合精度无法演示")

# ========== 2. 定义一个简单的模型 ==========

class SimpleModel(nn.Module):
    def __init__(self, d_model = 512, num_layers = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return self.norm(x)
    
batch_size = 32
seq_len = 128
d_model = 512

x = torch.randn(batch_size, seq_len, d_model).to(device)
target = torch.randn(batch_size, seq_len, d_model).to(device)

# ========== 4. fp32 训练 ==========
print("\n" + "=" * 50)
print("fp32 训练")
print("=" * 50)

model_fp32 = SimpleModel().to(device)
optimizer_fp32 = torch.optim.Adam(model_fp32.parameters(), lr = 1e-4)
loss_fn = nn.MSELoss()

# 显存统计
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

start = time.time()
for step in range(20):
    y_pred = model_fp32(x)
    loss = loss_fn(y_pred, target)
    optimizer_fp32.zero_grad()
    loss.backward()
    optimizer_fp32.step()
time_fp32 = time.time() - start

if device.type == "cuda":
    mem_fp32 = torch.cuda.max_memory_allocated() / 1e6
    print(f"峰值显存：{mem_fp32:.1f} MB")
print(f"20 步耗时：{time_fp32:.2f} 秒")
print(f"最终 loss：{loss.item():.4f}")

del model_fp32, optimizer_fp32
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# ========== 5. AMP 训练 ==========
print("\n" + "=" * 50)
print("AMP 混合精度训练")
print("=" * 50)

model_amp = SimpleModel().to(device)
optimizer_amp = torch.optim.Adam(model_amp.parameters(), lr = 1e-4)
scaler = torch.amp.GradScaler("cuda")
if device.type == "cuda" :
    torch.cuda.reset_peak_memory_stats()
start = time.time()
for step in range(20):
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        y_pred = model_amp(x)
        loss = loss_fn(y_pred, target)
        optimizer_amp.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer_amp)
        scaler.update()

time_amp = time.time() - start
if device.type == "cuda":
    mem_amp = torch.cuda.max_memory_allocated() / 1e6
    print(f"峰值显存：{mem_amp:.1f} MB")
print(f"20 步耗时：{time_amp:.2f} 秒")
print(f"最终 loss：{loss.item():.4f}")

