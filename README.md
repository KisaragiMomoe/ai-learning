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

## 总计划：
# 大模型算法工程师 · 学习计划与进度文档

> 适合保存到本地，或贴到 GitHub。以后对话太长时，把这份文档贴回来，就能快速恢复上下文。

---

## 一、 我的背景与目标

- **身份**：大一学生，会写 Python。
- **目标**：大模型算法工程师方向。
- **路线**：先打基础 → 做小项目 → 学 Transformer → 大模型微调 → 完整算法项目。
- **原则**：不抄代码，不收藏课程，每学一个概念必须动手跑实验。

---

## 二、 当前进度（已完成 Day 1 – Day 15）

### 第一周：Python 与数据处理
| 天数 | 内容 | 产出 |
|---|---|---|
| Day 1 | 环境搭建、Git、第一个 Python 文件 | `day1_hello.py` |
| Day 2 | 函数、类、文件读写 | `day2_python_review.py` |
| Day 3 | NumPy 矩阵运算、softmax、Attention 原理 | `day3_numpy.py` |
| Day 4 | Pandas 读写 CSV、筛选、统计 | `day4_pandas.py` |
| Day 5 | Matplotlib 画图、模拟 loss 曲线 | `day5_plot.py` |
| Day 6 | 学生成绩分析小项目（类 + 文件 + 报告） | `day6_project/` |
| Day 7 | 第一周复盘、写 README、推 GitHub | `README.md` |

### 第二周：机器学习与神经网络
| 天数 | 内容 | 产出 |
|---|---|---|
| Day 8 | NumPy 手写线性回归、偏导数、梯度下降 | `day8_linear_regression.py` |
| Day 9 | Mini-batch 梯度下降 | `day9_mini_batch_gd.py` |
| Day 10 | 过拟合、交叉熵、L2 正则化、Dropout | `day10_overfitting.py` |
| Day 11 | PyTorch 重写线性回归、nn.Linear、自动求导 | `day11_linear_regression_torch.py` |
| Day 12 | 房价预测完整项目（归一化、训练/测试划分、模型保存） | `day12_house_price.py` |
| Day 12.5 | 股票预测玩具项目（时间切分、IC、回测、手续费） | `day12_5_stock.py` |
| Day 13 | 第二周复盘、更新 README | `README.md` |

### 第三周（进行中）
| 天数 | 内容 | 状态 |
|---|---|---|
| Day 14 | MNIST 手写数字分类（全连接网络） | ✅ 完成 |
| Day 15 | CNN 卷积神经网络（准确率 99%+） | ✅ 完成 |
| Day 16 | RNN 与序列建模 | ⬜ 待做 |
| Day 17 | Transformer 逐层拆解（上） | ⬜ 待做 |
| Day 18 | Transformer 逐层拆解（中） | ⬜ 待做 |
| Day 19 | 手写 Multi-Head Attention | ⬜ 待做 |
| Day 20 | Mini GPT 复现 | ⬜ 待做 |

---

## 三、 后续计划（Day 16 之后）

### 阶段 A：第三周剩余（Day 16 – Day 20）
- **Day 16**：RNN、LSTM 直觉，用 PyTorch 做序列预测。
- **Day 17–18**：Transformer 结构拆解（Self-Attention、Multi-Head、位置编码、残差、LayerNorm、FFN）。
- **Day 19**：手写 Multi-Head Attention，不调库。
- **Day 20**：复现 nanoGPT，训练字符级小 GPT。

### 阶段 B：集中补数学（5 天）
- 线性代数：向量、矩阵、矩阵乘法、转置、范数、特征值。
- 微积分：导数、偏导数、链式法则、梯度。
- 概率统计：期望、方差、条件概率、贝叶斯、最大似然。
- 信息论：熵、交叉熵、KL 散度。
- 把数学和代码对应起来（softmax、交叉熵、梯度下降）。

### 阶段 C：大模型核心算法（预计 4–6 周）
- 预训练与数据 pipeline。
- SFT、LoRA、QLoRA、PEFT。
- 对齐：Reward Model、RLHF、DPO。
- 推理优化：KV Cache、量化、蒸馏、vLLM。
- 完整项目：选开源小模型（Qwen2.5-0.5B），做数据清洗、SFT、LoRA、评测、部署、对比实验。

### 阶段 D：求职准备
- 整理 GitHub 项目，写技术报告。
- 刷面试题：Attention、LoRA、SFT、DPO、量化、KV Cache。
- 参加竞赛或实验室项目。

---

## 四、 已掌握的核心概念清单

### Python 与工具
- 虚拟环境、Git、GitHub、README、requirements.txt
- 函数、类、`self`、`__init__`、文件读写（`with open`）
- JSON / JSONL / CSV 格式

### 数学与深度学习基础
- 矩阵乘法、转置、softmax、广播、keepdims
- 前向传播、反向传播、链式法则、偏导数
- 梯度下降、学习率、Mini-batch、batch_size
- 损失函数：MSE、交叉熵
- 过拟合、欠拟合、L2 正则化、Dropout、早停
- 归一化（Z-score）、训练集/测试集划分

### PyTorch
- `nn.Linear`、`nn.Sequential`、`nn.ReLU`、`nn.Flatten`
- `nn.Conv2d`、`nn.MaxPool2d`
- `nn.MSELoss`、`nn.CrossEntropyLoss`
- `optim.SGD`、`optim.Adam`、`weight_decay`
- `loss.backward()`、`optimizer.step()`、`zero_grad()`
- `model.train()` / `model.eval()`
- `DataLoader`、`batch_size`、`shuffle`
- `torch.save` / `load_state_dict`
- `argmax(dim=1)`、准确率计算

### 大模型核心（已接触）
- Attention 的 Q、K、V、Mask、缩放点积
- 生成文本时的自回归、KV Cache（直觉）
- 量化、蒸馏、推理加速（概念）
- SFT、LoRA、DPO（概念）

### 量化投资（玩具级）
- 时间切分、前视偏差、IC、回测、交易成本
- 买入持有基准

---

## 五、 踩过的坑与经验

| 坑 | 教训 |
|---|---|
| 学习率太大 → loss 变成 nan | 数据要归一化，学习率要调小 |
| 随机切分股票数据 → 回测虚假地好 | 时间序列必须按时间切分 |
| 忘记 `optimizer.zero_grad()` | 梯度会累加，必须清零 |
| `nn.MSELoss` 漏写括号 | 类和实例的区别 |
| `numpy()` 漏写括号 | 方法本身和方法调用的区别 |
| 特征尺度差异大 | 归一化，否则大特征主导 |
| 虚拟环境没激活 | 终端前面要有 `(.venv)` |
| Git 推送冲突 | 先 `git pull --rebase`，再 `push` |

---

## 六、 学习习惯

- 每天至少 1 小时看概念 + 1 小时写代码。
- 每周末复盘，更新 README，提交 GitHub。
- 每学一个概念，用自己的话写笔记。
- 不复制粘贴，手动敲代码。
- 遇到报错先读错误信息，再搜索，最后问人。

---

## 七、 下一步行动

- [ ] 继续 Day 16：RNN 与序列建模。
- [ ] 完成 Day 17–20：Transformer 与 Mini GPT。
- [ ] 集中 5 天补数学。
- [ ] 进入大模型微调阶段。

---

**保存这份文档。以后如果对话太长，把这份文档贴回来，我就能立刻恢复上下文。**