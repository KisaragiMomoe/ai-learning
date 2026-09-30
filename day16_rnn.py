import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
t = np.linspace(0, 100, 1000)
data = np.sin(t) + np.random.randn(1000) * 0.1

seq_len = 20
x, y = [], []
for i in range(len(data) - seq_len):
    x.append(data[i:i + seq_len])
    y.append(data[i + seq_len])
x = np.array(x, dtype = np.float32).reshape(-1, seq_len, 1)
y = np.array(y, dtype = np.float32).reshape(-1, 1)
print(f"X 形状：{x.shape} | y 形状：{y.shape}")

split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]
x_train_t = torch.tensor(x_train)
y_train_t = torch.tensor(y_train)
x_test_t = torch.tensor(x_test)
y_test_t = torch.tensor(y_test)

class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size = 1, hidden_size = 32, num_layers = 1, batch_first = True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, h_n = self.rnn(x)
        last_out = out[:, -1, :]
        return self.fc(last_out) 

model = RNNModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

epochs, train_losses, test_losses = 100, [], []
for epoch in range(epochs):
    model.train()
    y_pred = model(x_train_t)
    loss = loss_fn(y_pred, y_train_t)
    optimizer.zero_grad()
    loss.backward()
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
plt.title("正弦波预测对比")
plt.xlabel("时间步")
plt.ylabel("值")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("day16_rnn.png")
plt.show()

print("\n图片已保存：day16_rnn.png")