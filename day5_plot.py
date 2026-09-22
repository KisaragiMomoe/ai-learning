import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("students_with_avg.csv", encoding = "utf-8-sig")
# plt.figure(figsize = (8, 5))
# plt.bar(df["name"], df["avg"])
# plt.title("Average Score by student")
# plt.xlabel("Student")
# plt.ylabel("Average Score")
# plt.savefig("avg_bar.png")
# plt.show()

# df_sorted = df.sort_values("avg")
# plt.figure(figsize = (8, 5))
# plt.plot(df_sorted["name"], df_sorted["avg"], marker = "o")
# plt.title("Average Score Trend")
# plt.xlabel("Student")
# plt.ylabel("Average Score")
# plt.grid(True)
# plt.savefig("avg_line.png")
# plt.show()

# 4. 模拟训练 loss 曲线
# epochs = list(range(1, 11))          # 1 到 10
# losses = [2.5, 2.0, 1.6, 1.2, 0.9, 0.7, 0.55, 0.45, 0.38, 0.32]

# plt.figure(figsize=(8, 5))
# plt.plot(epochs, losses, marker="o", color="red")
# plt.title("Simulated Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.savefig("simulated_loss.png")
# plt.show()

plt.rcParams["font.sans-serif"] = ["SimHei"]   # Windows 用黑体
plt.rcParams["axes.unicode_minus"] = False     # 解决负号显示问题

# 5. 散点图：数学 vs 英语
plt.figure(figsize=(8, 5))
plt.scatter(df["math"], df["english"], color="green")
plt.title("数学 vs 英语")
plt.xlabel("Math")
plt.ylabel("English")
plt.grid(True)
plt.savefig("math_english_scatter.png")
plt.show()