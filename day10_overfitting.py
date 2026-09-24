import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
X = np.linspace(0, 1, 20).reshape(-1, 1)
Y = np.sin(2 * np.pi * X) + np.random.randn(20, 1) * 0.2

X_tensor = torch.tensor(X, dtype = torch.float32)
Y_tensor = torch.tensor(Y, dtype = torch.float32)

class BigModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_model(weight_decay = 0.0, epochs = 2000):
    model = BigModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = weight_decay)
    loss_fn = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        Y_pred = model(X_tensor)
        loss = loss_fn(Y_pred, Y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return model, losses

model_no_reg, losses_no_reg = train_model(weight_decay=0.0)
model_with_reg, losses_with_reg = train_model(weight_decay=0.01)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, Y, color="black", label="真实数据（含噪声）")
X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
X_plot_tensor = torch.tensor(X_plot, dtype=torch.float32)

with torch.no_grad():
    y_plot_no_reg = model_no_reg(X_plot_tensor).numpy()
    y_plot_with_reg = model_with_reg(X_plot_tensor).numpy()

plt.plot(X_plot, y_plot_no_reg, color="red", label="无正则化（过拟合）")
plt.plot(X_plot, y_plot_with_reg, color="green", label="L2 正则化（平滑）")
plt.title("模型拟合效果对比")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(losses_no_reg, color="red", label="无正则化 Loss")
plt.plot(losses_with_reg, color="green", label="L2 正则化 Loss")
plt.title("训练 Loss 对比")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("day10_overfitting.png")
plt.show()