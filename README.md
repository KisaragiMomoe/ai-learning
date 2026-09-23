# AI 学习第一周项目

## 项目简介
这是我学习大模型算法的第一周项目。通过这个项目，我掌握了 Python 基础、NumPy 矩阵运算、Pandas 数据处理、Matplotlib 可视化，并完成了一个完整的学生成绩分析小项目。

## 项目结构
```text
ai_learning/
  day1_hello.py          # 第一个 Python 文件
  day2_review.py  # 函数和类复习
  day3_numpy.py          # NumPy 矩阵运算和 softmax
  day4_pandas.py         # Pandas 读写 CSV 和筛选统计
  day5_plot.py           # Matplotlib 画柱状图、折线图、散点图
  day6_project/          # 完整小项目
    main.py              # 主程序
    students_with_avg.csv
    avg_bar.png
    report.txt
  README.md
```

## 如何运行
1. 创建并激活虚拟环境：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # Mac/Linux
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 运行 Day6 项目：
   ```bash
   cd day6_project
   python main.py
   ```

## 我学到了什么
- Python 函数、类、文件读写
- NumPy 矩阵运算、转置、softmax 实现
- Attention 的 Q、K、V 计算和 Mask 机制
- Pandas 读写 CSV、筛选、统计、排序
- Matplotlib 画柱状图、折线图、散点图、模拟 loss 曲线
- 用类组织代码，链式调用

## 下一周计划
- 学习数学基础（线性代数、概率统计）
- 用 NumPy 实现线性回归
- 理解梯度下降
