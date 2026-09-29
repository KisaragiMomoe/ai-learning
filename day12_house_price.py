import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
n = 500
area = np.random.uniform(50, 200, n)
rooms = np.random.randint(1, 6, n)
age = np.random.uniform(0, 30, n)
price = 2 * area + 5 * rooms - 3 * age + 20 + np.random.randn(n) * 10

x = np.stack([area, rooms, age], axis = 1).astype(np.float32)
y = price.astype(np.float32).reshape(-1, 1)

x_mean = x.mean(axis = 0)
y_mean = y.mean(axis = 0)
x_std = x.std(axis = 0)
y_std = y.std(axis = 0)
x = (x - x_mean) / x_std
y = (y - y_mean) / y_std

indices = np.random.permutation(n)
split = 400
train_idx, test_idx = indices[:split], indices[split:]

x_train = torch.tensor(x[train_idx])
y_train = torch.tensor(y[train_idx])
x_test = torch.tensor(x[test_idx])
y_test = torch.tensor(y[test_idx])

print(f"训练集：{x_train.shape[0]} 条 | 测试集：{x_test.shape[0]} 条")

class HouseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = HouseModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0001)

epochs = 500
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        test_loss = loss_fn(y_test_pred, y_test)
        test_losses.append(test_loss.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

plt.figure(figsize=(12, 5))

# 左图：Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="训练 Loss", color="blue")
plt.plot(test_losses, label="测试 Loss", color="red")
plt.title("训练与测试 Loss 曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

model.eval()
with torch.no_grad():
    y_pred_test = model(x_test).numpy()

y_pred_origin = y_pred_test * y_std + y_mean
y_test_origin = y_test.numpy() * y_std + y_mean

plt.subplot(1, 2, 2)
plt.scatter(y_test_origin, y_pred_origin, alpha=0.5, color="green")
plt.plot([y_test_origin.min(), y_test_origin.max()],
         [y_test_origin.min(), y_test_origin.max()],
         color="red", linestyle="--", label="理想预测线")
plt.title("测试集：预测值 vs 真实值")
plt.xlabel("真实房价")
plt.ylabel("预测房价")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("day12_house_price.png")
plt.show()

torch.save(model.state_dict(), "day12_house_model.pth")
print("\n模型已保存为 day12_house_model.pth")