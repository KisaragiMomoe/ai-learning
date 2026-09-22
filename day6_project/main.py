import os
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class ScoreAnalyzer:
    def __init__(self, data):
        self.df = pd.DataFrame(data)

    def clean(self):
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        return self

    def add_metrics(self):
        self.df["total"] = self.df[["chinese", "math", "english"]].sum(axis = 1)
        self.df["avg"] = self.df[["chinese", "math", "english"]].mean(axis = 1)
        self.df["rank"] = self.df["avg"].rank(ascending = 1).astype(int)
        return self

    def save_csv(self, path):
        self.df.to_csv(path, index = False, encoding = "utf-8")
        print(f"csv saved to: {path}")

    def plot_avg(self, path):
        plt.figure(figsize=(8, 5))
        plt.bar(self.df["name"], self.df["avg"])
        plt.title("学生平均分")
        plt.xlabel("学生")
        plt.ylabel("平均分")
        plt.savefig(path)
        plt.close()
        print(f"image saved to: {path}")

    def write_report(self, path):
        with open(path, "w", encoding = "utf-8") as f:
            f.write("学生成绩分析报告\n")
            f.write("=" * 30 + "\n")
            f.write(self.df.to_string(index=False))
            f.write("\n\n")
            f.write(f"班级平均分：{self.df['avg'].mean():.2f}\n")
            f.write(f"最高平均分：{self.df['avg'].max():.2f}\n")
            f.write(f"最低平均分：{self.df['avg'].min():.2f}\n")
            f.write("\n平均分大于 80 的学生：\n")
            good = self.df[self.df["avg"] > 80]
            f.write(good.to_string(index=False))
        print(f"已保存报告：{path}")


def main():
    data = {
        "name": ["Alice", "Bob", "Charlie", "David", "Eva"],
        "chinese": [85, 92, 65, 88, 95],
        "math": [78, 88, 70, 90, 80],
        "english": [90, 85, 60, 92, 88]
    }

    analyzer = ScoreAnalyzer(data)
    analyzer.clean().add_metrics()
    analyzer.save_csv("students_with_avg.csv")
    analyzer.plot_avg("avg_bar.png")
    analyzer.write_report("report.txt")

if __name__ == "__main__":
    main()
