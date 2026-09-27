import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
x = np.array([50, 60, 70, 80, 90, 100, 110, 120], dtype = np.float32)
y = x * 1.5 + 20 + np.random.randn(len(x)) * 5
y = y.astype(np.float32)

x_tensor = torch.tensor(x).reshape(-1, 1)
y_tensor = torch.tensor(y).reshape(-1, 1)

# model = nn.Linear(1, 1)

model = nn.Sequential(
    nn.Linear(1, 16), nn.ReLU(),
    nn.Linear(16, 1)
)


loss_fn = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
epochs = 10000
loss_history = []

for epoch in range(epochs):
    y_pred = model(x_tensor)
    loss = loss_fn(y_pred, y_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    # if epoch % 100 == 0:
        # w = model.weight.item()
        # b = model.bias.item()
        # print(f"Epoch {epoch:4d} | Loss: {loss.item():.2f} | w: {w:.4f} | b: {b:.4f}")
# w_final = model.weight.item()
# b_final = model.bias.item()

# print(f"\n训练完成！最终 w = {w_final:.4f}, b = {b_final:.4f}")
# print(f"真实关系：y = 1.5 * x + 20")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(x, y)
plt.plot(x, model(x_tensor).detach().numpy(), color="red", label="拟合直线")
plt.title("PyTorch 线性回归拟合")
plt.xlabel("面积（平米）")
plt.ylabel("房价（万元）")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss_history, color="green")
plt.title("训练 Loss 曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("day11_linear_regression_torch.png")
plt.show()