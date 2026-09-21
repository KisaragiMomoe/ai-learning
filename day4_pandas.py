import pandas as pd

data = {
    "name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "math": [85, 92, 65, 88, 95],
    "english": [78, 88, 70, 90, 80],
    "python": [90, 85, 60, 92, 88]
}

df = pd.DataFrame(data)

# print(df)

df.to_csv("students.csv", index=False, encoding = "utf-8-sig")

df2 = pd.read_csv("students.csv", encoding = "utf-8-sig")

print(df2.head())

print(df2.shape)

print(df2.columns.tolist())

print(df2.describe())

print(df2[df2["math"] > 85])

print(df2[(df2["math"] > 85) & (df2["english"] > 80)])

print(df2[df2["name"].isin(["Alice", "Bob"])])

# 12. 计算每个学生的平均分
df2["avg"] = df2[["math", "english", "python"]].mean(axis=1)
print("\n加了平均分：")
print(df2)

# 13. 按平均分从高到低排序
df_sorted = df2.sort_values("avg", ascending=False)
print("\n按平均分排序：")
print(df_sorted)

# 14. 计算每科平均分
print("\n每科平均分：")
print(df2[["math", "english", "python"]].mean())

# 15. 计算每科最高分
print("\n每科最高分：")
print(df2[["math", "english", "python"]].max())

df2.to_csv("students_with_avg.csv", index=False, encoding="utf-8-sig")
print("\n已保存 students_with_avg.csv")
