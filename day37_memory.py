import torch
import torch.nn as nn
import time
from torch.utils.checkpoint import checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备：{device}")
if device.type == "cuda":
    print(f"GPU：{torch.cuda.get_device_name(0)}")

class BigModel(nn.Module):
    def __init__(self, d_model = 512, num_layers = 12):
        super().__init__()
        self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model)
                )
                for _ in range(num_layers)
            ])
        self.norm = nn.LayerNorm(d_model)
        self.use_checkpoint = False

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = x + checkpoint(layer, x, use_reentrant=False)
            else:
                x = x + layer(x)
        return self.norm(x)

batch_size = 32
seq_len = 128
d_model = 512

x = torch.randn(batch_size, seq_len, d_model).to(device)
target = torch.randn(batch_size, seq_len, d_model).to(device)
loss_fn = nn.MSELoss()

def train_config(name, use_amp = False, use_checkpoint = False, accum_step = 1):
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    model = BigModel().to(device)
    model.use_checkpoint = use_checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model.train()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    start = time.time()
    for step in range(20):
        if use_amp:
            with torch.cuda.amp.autocast():
                y_pred = model(x)
                loss = loss_fn(y_pred, target) / accum_step
            scaler.scale(loss).backward()
            if (step + 1) % accum_step == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            y_pred = model(x)
            loss = loss_fn(y_pred, target) / accum_step
            loss.backward()
            if (step + 1) % accum_step == 0:
                optimizer.step()
                optimizer.zero_grad()
    elapsed = time.time() - start
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"峰值显存：{mem:.1f} MB")
    print(f"20 步耗时：{elapsed:.2f} 秒")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return elapsed
train_config("1. 基线（fp32，无优化）")
train_config("2. + 梯度检查点", use_checkpoint=True)
if device.type == "cuda":
    train_config("3. + 混合精度", use_amp=True)
    train_config("4. + 混合精度 + 梯度检查点", use_amp=True, use_checkpoint=True)
else:
    print("\n没有 GPU，跳过混合精度实验")