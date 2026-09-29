# AI 学习项目：从 Python 到神经网络

这是我学习大模型算法的完整记录。两周内，我从 Python 基础出发，手写了梯度下降，用 PyTorch 重写了线性回归，并完成了两个完整项目：房价预测和股票预测。

## 项目结构

```text
ai_learning/
  # 第一周：Python 基础与数据处理
  day1_hello.py                    # 第一个 Python 文件
  day2_python_review.py            # 函数和类复习
  day3_numpy.py                    # NumPy 矩阵运算与 softmax
  day4_pandas.py                   # Pandas 读写 CSV
  day5_plot.py                     # Matplotlib 画图
  day6_project/                    # 学生成绩分析小项目

  # 第二周：机器学习与神经网络
  day8_linear_regression.py        # NumPy 手写线性回归
  day9_mini_batch_gd.py            # Mini-batch 梯度下降
  day10_overfitting.py             # 过拟合与正则化
  day11_linear_regression_torch.py # PyTorch 重写线性回归
  day12_house_price.py             # 房价预测完整项目
  day12_5_stock.py                 # 股票预测玩具项目

  requirements.txt                 # 依赖清单
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

3. 运行任意文件，例如：
   ```bash
   python day12_house_price.py
   ```

## 我学到了什么

### 第一周：Python 与数据处理
- Python 函数、类、文件读写
- NumPy 矩阵运算、转置、softmax 实现
- Attention 的 Q、K、V 计算和 Mask 机制
- Pandas 读写 CSV、筛选、统计、排序
- Matplotlib 画柱状图、折线图、散点图

### 第二周：机器学习与神经网络
- 线性回归的数学原理：前向传播、MSE、偏导数、梯度下降
- Mini-batch 梯度下降与 batch_size 的影响
- 交叉熵、过拟合、L2 正则化、Dropout
- PyTorch 核心：nn.Linear、自动求导、optimizer
- 完整项目流程：数据处理、归一化、训练/测试划分、可视化、模型保存
- 量化入门：时间切分、IC、回测、交易成本

## 核心项目

### 1. 房价预测（day12_house_price.py）
- 3 个特征：面积、房间数、房龄
- 神经网络：3 层 MLP
- 训练集/测试集 8:2 划分
- 数据标准化、L2 正则化、模型保存

### 2. 股票预测（day12_5_stock.py）
- 6 个技术因子
- 按时间切分（避免前视偏差）
- IC 评估、回测、扣手续费
- 对比买入持有基准

## 下一阶段计划
- 数学基础：微积分、概率统计、线性代数
- 深度学习核心：CNN、RNN、Transformer
- 大模型微调：SFT、LoRA、DPO
