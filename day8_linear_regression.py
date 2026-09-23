import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

X = np.array([50, 60, 70, 80, 90, 100, 110, 120])
np.random.seed(42)
y = X * 1.5 + 20 + np.random.randint(len(X)) * 5

w = 0.0
b = 0.0
learning_rate = 0.000001
epochs = 1000
loss_history = []

for epoch in range(epochs):
    y_pred = w * X + b
    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)

    dw = np.mean(2 * (y_pred - y) * X)
    db = np.mean(2 * (y_pred - y))

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if (epoch % 100 == 0):
        print(f"Epoch {epoch:4d} | Loss: {loss:.2f} | w: {w:.4f} | b: {b:.4f}")

print(f"\n训练完成！最终 w = {w:.4f}, b = {b:.4f}")
print(f"真实关系大概是：y = 1.5 * x + 20")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color="blue", label="真实数据")
plt.plot(X, w * X + b, color="red", label="拟合直线")
plt.title("面积 vs 房价")
plt.xlabel("面积（平米）")
plt.ylabel("房价（万元）")
plt.legend()
plt.grid(True)

# 子图2：Loss 曲线
plt.subplot(1, 2, 2)
plt.plot(loss_history, color="green")
plt.title("训练 Loss 曲线")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("day8_linear_regression.png")
plt.show()

print("\n图片已保存：day8_linear_regression.png")
