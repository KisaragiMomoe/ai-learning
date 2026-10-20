"""
Day 39: 用 Accelerate 改造训练脚本
同一套代码，适配 CPU / 单 GPU / 多 GPU。
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

# ========== 1. 初始化 Accelerator ==========
accelerator = Accelerator(mixed_precision="bf16")
print(f"混合精度：{accelerator.mixed_precision}")  # 这次绝对会显示 bf16


# 打印环境信息
print(f"进程数：{accelerator.num_processes}")
print(f"当前进程：{accelerator.process_index}")
print(f"设备：{accelerator.device}")
print(f"混合精度：{accelerator.mixed_precision}")


# ========== 2. 定义数据集 ==========
class StockDataset(Dataset):
    def __init__(self, n_samples=1000, seq_len=20, n_features=32):
        self.X = torch.randn(n_samples, seq_len, n_features)
        self.y = torch.randn(n_samples, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ========== 3. 定义模型 ==========
class SimpleModel(nn.Module):
    def __init__(self, n_features=32, d_model=128):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)   # 池化
        return self.output_proj(x)


# ========== 4. 准备数据 ==========
train_dataset = StockDataset(n_samples=2000)
val_dataset = StockDataset(n_samples=400)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# ========== 5. 创建模型、优化器、损失 ==========
model = SimpleModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
loss_fn = nn.MSELoss()

model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

epochs = 5

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        accelerator.backward(loss)     # 替代 loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 只在主进程打印
    if accelerator.is_main_process:
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f}")

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

    # 多卡时，所有进程的 loss 要聚合
    val_loss_tensor = torch.tensor(val_loss / len(val_loader), device=accelerator.device)
    val_loss_tensor = accelerator.gather(val_loss_tensor).mean()

    if accelerator.is_main_process:
        print(f"Epoch {epoch+1}/{epochs} | Val Loss: {val_loss_tensor.item():.6f}")


# ========== 8. 保存模型（只在主进程） ==========
if accelerator.is_main_process:
    save_dir = "./day39_output"
    os.makedirs(save_dir, exist_ok=True)

    # 多卡时，模型被 DDP 包装，要先用 unwrap_model 解包
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "model.pth"))
    print(f"模型已保存到：{save_dir}")


# ========== 9. 清理 ==========
accelerator.end_training()
print("训练完成")