# 大模型算法工程师 · 学习记录与计划

> 本仓库记录我从零开始学习大模型算法的完整历程。  
> 包含每周学习内容、代码实现、实验报告和学习计划。  
> 适合作为学习档案，也方便随时恢复进度。

---

## 一、 项目简介

- **身份**：大一学生，会写 Python。
- **目标**：大模型算法工程师方向。
- **路线**：先打基础 → 做小项目 → 学 Transformer → 大模型微调 → 完整算法项目。
- **原则**：不抄代码，不收藏课程，每学一个概念必须动手跑实验。

---

## 二、 学习进度总览

| 阶段 | 周次 | 主题 | 状态 |
|---|---|---|---|
| 第一阶段 | 第 1 周 | Python、NumPy、Pandas、Matplotlib、Git | ✅ 完成 |
| 第一阶段 | 第 2 周 | 线性回归、梯度下降、PyTorch、房价预测、股票玩具项目 | ✅ 完成 |
| 第一阶段 | 第 3 周 | MNIST、CNN、RNN、Transformer、Mini GPT | ✅ 完成 |
| 第二阶段 | 第 4 周 | 集中补数学：线代、微积分、概率、信息论 | ✅ 完成 |
| 第三阶段 | 第 5 周 | 大模型微调实战：HuggingFace、SFT、LoRA、DPO、推理优化 | 🔄 进行中 |
| 第三阶段 | 第 6-8 周 | 完整微调项目、评测、部署、报告 | ⬜ 待开始 |
| 第四阶段 | 第 9-12 周 | 进阶：RLHF、分布式训练、论文复现、求职准备 | ⬜ 待开始 |

---

## 三、 已完成内容（按周）

### 第 1 周：Python 与数据处理
- **Day 1**：环境搭建、Git、第一个 Python 文件
- **Day 2**：函数、类、文件读写
- **Day 3**：NumPy 矩阵运算、softmax、Attention 原理
- **Day 4**：Pandas 读写 CSV、筛选、统计
- **Day 5**：Matplotlib 画图、模拟 loss 曲线
- **Day 6**：学生成绩分析小项目（类 + 文件 + 报告）
- **Day 7**：第一周复盘、写 README、推 GitHub

**产出**：`day1_hello.py` ~ `day6_project/`

### 第 2 周：机器学习与神经网络
- **Day 8**：NumPy 手写线性回归、偏导数、梯度下降
- **Day 9**：Mini-batch 梯度下降
- **Day 10**：过拟合、交叉熵、L2 正则化、Dropout
- **Day 11**：PyTorch 重写线性回归、nn.Linear、自动求导
- **Day 12**：房价预测完整项目（归一化、训练/测试划分、模型保存）
- **Day 12.5**：股票预测玩具项目（时间切分、IC、回测、手续费）
- **Day 13**：第二周复盘、更新 README

**产出**：`day8_linear_regression.py` ~ `day12_5_stock.py`

### 第 3 周：深度学习与 Transformer
- **Day 14**：MNIST 手写数字分类（全连接网络）
- **Day 15**：CNN 卷积神经网络（准确率 99%+）
- **Day 16**：RNN 与序列建模（正弦波预测）
- **Day 17**：Transformer 逐层拆解（上）：Multi-Head Attention、位置编码
- **Day 18**：完整 Transformer Encoder 组装
- **Day 19**：训练一个小 Transformer
- **Day 20**：Mini GPT 复现，训练字符级语言模型

**产出**：`day14_mnist.py` ~ `day20_mini_gpt.py`

### 第 4 周：集中补数学
- **Day 21**：线性代数（向量、矩阵、转置、广播、范数、点积、特征值）
- **Day 22**：微积分（导数、偏导数、链式法则、梯度、GELU、ReLU）
- **Day 23**：概率统计（期望、方差、条件概率、贝叶斯、MLE、采样）
- **Day 24**：信息论（信息量、熵、交叉熵、KL 散度、困惑度、NLL）
- **Day 25**：数学与代码对应 + 第四周总结

**产出**：`day21_linalg.py` ~ `day24_information.py`

### 第 5 周（进行中）：大模型微调实战
- **Day 26**：HuggingFace 生态入门，加载 Qwen2.5-0.5B 并推理
- **Day 27**：SFT 监督微调（待完成）
- **Day 28**：LoRA 与 QLoRA（待完成）
- **Day 29**：DPO 对齐（待完成）
- **Day 30**：推理优化（KV Cache、量化、vLLM）（待完成）
- **Day 31-35**：完整微调项目（待完成）

**产出**：`day26_huggingface.py` 等

---

## 四、 项目结构

```text
ai_learning/
├── day1_hello.py
├── day2_python_review.py
├── day2_report.txt
├── day3_numpy.py
├── day4_pandas.py
├── day5_plot.py
├── day6_project/
│   ├── main.py
│   ├── students_with_avg.csv
│   ├── avg_bar.png
│   └── report.txt
├── day8_linear_regression.py
├── day9_mini_batch_gd.py
├── day10_overfitting.py
├── day11_linear_regression_torch.py
├── day12_house_price.py
├── day12_5_stock.py
├── day14_mnist.py
├── day15_cnn.py
├── day16_rnn.py
├── day17_transformer.py
├── day18_transformer.py
├── day19_train_transformer.py
├── day20_mini_gpt.py
├── day26_huggingface.py
├── requirements.txt
└── README.md
```

---

## 五、 如何运行

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
   python day20_mini_gpt.py
   ```

---

## 六、 已掌握技能

### Python 与工具
- 虚拟环境、Git、GitHub、README、requirements.txt
- 函数、类、`self`、`__init__`、文件读写
- JSON / JSONL / CSV 格式

### 数学与深度学习基础
- 矩阵乘法、转置、softmax、广播、keepdims
- 前向传播、反向传播、链式法则、偏导数
- 梯度下降、学习率、Mini-batch、batch_size
- 损失函数：MSE、交叉熵、负对数似然
- 过拟合、欠拟合、L2 正则化、Dropout、早停
- 归一化（Z-score）、训练集/测试集划分
- 信息论：熵、交叉熵、KL 散度、困惑度
- 概率：期望、方差、条件概率、贝叶斯、MLE、采样

### PyTorch
- `nn.Linear`、`nn.Sequential`、`nn.ReLU`、`nn.GELU`、`nn.Flatten`
- `nn.Conv2d`、`nn.MaxPool2d`
- `nn.RNN`、`nn.LSTM`
- `nn.MultiheadAttention`、`nn.TransformerEncoderLayer`
- `nn.Embedding`、`nn.LayerNorm`、`nn.Dropout`
- `nn.MSELoss`、`nn.CrossEntropyLoss`
- `optim.SGD`、`optim.Adam`、`optim.AdamW`、`weight_decay`
- `loss.backward()`、`optimizer.step()`、`zero_grad()`
- `model.train()` / `model.eval()`
- `DataLoader`、`batch_size`、`shuffle`
- `torch.save` / `load_state_dict`
- `argmax(dim=1)`、准确率计算
- `clip_grad_norm_`、梯度裁剪
- `torch.multinomial`、采样

### 大模型核心
- Attention 的 Q、K、V、Mask、缩放点积
- Multi-Head Attention、位置编码、残差连接、LayerNorm、FFN
- Transformer Encoder、Decoder、GPT 架构
- 自回归生成、KV Cache（直觉）
- 交叉熵、最大似然估计、DPO、KL 散度
- HuggingFace：AutoTokenizer、AutoModelForCausalLM、generate

### 量化投资（玩具级）
- 时间切分、前视偏差、IC、回测、交易成本
- 买入持有基准

---

## 七、 深化学习计划（第 5 周及以后）

### 第 5 周：大模型微调实战
- **Day 26**：HuggingFace 生态入门 ✅
- **Day 27**：SFT 监督微调，用指令数据微调 Qwen2.5-0.5B
- **Day 28**：LoRA 与 QLoRA，用少量参数微调大模型
- **Day 29**：DPO 对齐，用偏好数据优化模型
- **Day 30**：推理优化（KV Cache、量化、vLLM）
- **Day 31-35**：完整微调项目（数据清洗、训练、评测、部署、报告）

### 第 6-8 周：完整项目与评测
- 选一个垂直领域（如医疗问答、法律文书、游戏 NPC）
- 构造 500-1000 条 SFT 数据
- 对比全参微调、LoRA、QLoRA 的效果
- 做 DPO 对齐，对比 SFT-only 和 SFT+DPO
- 系统评测：指令遵循率、人工评分、延迟、显存
- 用 vLLM 部署，测吞吐和延迟
- 写技术报告和 GitHub README

### 第 9-12 周：进阶与求职
- **RLHF 深入**：Reward Model、PPO
- **分布式训练**：DeepSpeed、FSDP、混合精度
- **论文复现**：选一篇经典论文（如 LoRA、DPO）复现
- **竞赛/实验室**：参加 Kaggle、天池或加入学校实验室
- **面试准备**：整理高频面试题，模拟面试
- **简历与项目**：完善 GitHub，写技术博客

### 长期目标
- 大二：能独立微调小模型，参加一次比赛
- 大三：有大模型相关项目或论文，争取实习
- 大四：拿到大模型算法工程师 offer

---

## 八、 学习习惯

- 每天至少 1 小时看概念 + 1 小时写代码。
- 每周末复盘，更新 README，提交 GitHub。
- 每学一个概念，用自己的话写笔记。
- 不复制粘贴，手动敲代码。
- 遇到报错先读错误信息，再搜索，最后问人。

---

## 九、 踩过的坑与经验

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
| Mini GPT 过拟合 | 数据太少，模型太大，需要更多数据或正则化 |

---

## 十、 资源推荐

**基础**
- 吴恩达《Machine Learning》《Deep Learning》
- 李沐《动手学深度学习》
- CS224n、CS231n
- 3Blue1Brown 线性代数

**大模型**
- Karpathy：Zero to Hero、nanoGPT
- HuggingFace NLP Course / LLM Course
- DeepLearning.AI 大模型相关课程

**必读论文**
- Attention Is All You Need
- GPT-3
- InstructGPT
- LoRA
- QLoRA
- DPO
- PPO
- FlashAttention
- GPTQ / AWQ
- vLLM

**工具**
- PyTorch、Transformers、Datasets
- PEFT、TRL、bitsandbytes、Accelerate
- DeepSpeed、vLLM、llama.cpp
- WandB、GitHub

---

## 十一、 联系方式

- GitHub: kisaragi_momoe
- 邮箱: kisaragimomoe@outlook.com

---

> **最后更新**：Day 26 完成后  
> **下一里程碑**：完成 SFT + LoRA 微调，跑通第一个大模型微调实验。