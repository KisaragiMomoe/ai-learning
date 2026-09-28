import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import akshare as ak
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

print("正在下载数据...")
df = ak.stock_zh_a_hist(
    symbol = "600519",
    period = "daily",
    start_date = "20200101",
    end_date = "20241231",
    adjust = "qfq",
)
df = df.rename(columns = {"日期": "date", "收盘": "close", "成交量": "volume"})
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop = True)
print(f"数据量：{len(df)} 条，从 {df['date'].iloc[0].date()} 到 {df['date'].iloc[-1].date()}")
df["ret_1"] = df["close"].pct_change(1)
df["ret_5"] = df["close"].pct_change(5)
df["ret_20"] = df["close"].pct_change(20)
df["vol_5"] = df["ret_1"].rolling(5).std()
df["vol_20"] = df["ret_1"].rolling(20).std()
df["volume_change"] = df["volume"].pct_change(5)
df["target"] = df["close"].pct_change(1).shift(-1)
df = df.dropna().reset_index(drop=True)

feature_cols = ["ret_1", "ret_5", "ret_20", "vol_5", "vol_20", "volume_change"]
x = df[feature_cols].values.astype(np.float32)
y = df["target"].values.astype(np.float32).reshape(-1, 1)

n = len(x)
split = int(n * 0.7)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

x_mean, x_std = x_train.mean(axis = 0), x_train.std(axis = 0)
x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

x_train_t = torch.tensor(x_train)
y_train_t = torch.tensor(y_train)
x_test_t = torch.tensor(x_test)
y_test_t = torch.tensor(y_test)

class StockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = StockModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)

epochs = 200
train_losses = []
test_losses = []

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
        y_test_grad = model(x_test_t)
        test_loss = loss_fn(y_test_grad, y_test_t)
        test_losses.append(test_loss.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")

# ========== 预测与 IC 评估 ==========
model.eval()
with torch.no_grad():
    y_pred_test = model(x_test_t).numpy().flatten()

y_test_real = y_test.flatten()
ic = np.corrcoef(y_pred_test, y_test_real)[0, 1]
print(f"\n测试集 IC（信息系数）: {ic:.4f}")
print("（IC 接近 0 说明模型几乎没预测能力，>0.05 已经算不错）")

signal = (y_pred_test > 0).astype(int)
strategy_ret = signal * y_test_real

cost = 0.001
position_change = np.abs(np.diff(signal, prepend=0))
# strategy_ret_after_cost = strategy_ret - position_change * cost
strategy_ret_after_cost = strategy_ret   # 不扣费
cum_strategy = np.cumprod(1 + strategy_ret_after_cost) - 1
cum_buy_hold = np.cumprod(1 + y_test_real) - 1
print(f"\n策略累计收益（扣费后）: {cum_strategy[-1]*100:.2f}%")
print(f"买入持有累计收益:      {cum_buy_hold[-1]*100:.2f}%")

# ========== 10. 画图 ==========
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
plt.plot(cum_strategy * 100, label="策略（扣费后）", color="green")
plt.plot(cum_buy_hold * 100, label="买入持有", color="gray")
plt.title("测试集累计收益对比")
plt.xlabel("交易日")
plt.ylabel("累计收益 (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("day12_5_stock.png")
plt.show()

print("\n图片已保存：day12_5_stock.png")
