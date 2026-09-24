import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
X = np.linspace(0, 10, 200)
Y = 2.5 * X + 5 + np.random.randn(200) * 2

w = 0.0
b = 0.0
learning_rate = 0.001
epochs = 20
batch_size = 32
loss_history = []
n = len(X)

for epoch in range(epochs):
    indices = np.random.permutation(n)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    for i in range(0, n, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        y_pred = w * X_batch + b
        dw = np.mean(2 * (y_pred - Y_batch) * X_batch)
        db = np.mean(2 * (y_pred - Y_batch))
        w = w - learning_rate * dw
        b = b - learning_rate * db

    y_pred_all = w * X + b
    loss = np.mean((y_pred_all - Y) ** 2)
    loss_history.append(loss)
    print(f"Epoch {epoch+1:2d} | Loss: {loss:.4f} | w: {w:.4f} | b: {b:.4f}")

print(f"\n训练完成！最终 w = {w:.4f}, b = {b:.4f}")
print(f"真实关系：y = 2.5 * x + 5")

# 6. 画图
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, Y, color="blue", alpha=0.5, label="真实数据")
plt.plot(X, w * X + b, color="red", linewidth=2, label="拟合直线")
plt.title("面积 vs 房价（Mini-batch 拟合）")
plt.xlabel("面积（平米）")
plt.ylabel("房价（万元）")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss_history, color="green", marker="o")
plt.title("Mini-batch 训练 Loss 曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("day9_mini_batch_gd.png")
plt.show()