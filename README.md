# 🚀 大模型算法与应用全栈实战：从手写 Transformer 到赛博 Agent

本仓库记录了我在大模型领域的深度探索之旅。内容涵盖 **底层算法推导、大模型微调（SFT/LoRA/DPO）、推理加速（vLLM 源码剖析）、以及全栈 AI Agent 开发（Perplexity 搜索、赛博水群机器人）**。

---

## 🗺️ 知识图谱与学习路线

本项目的代码按照学习进度分为五大核心模块：

### 1. 深度学习与 Transformer 基础 (`day1` - `day20`)
从零开始手写核心算法，拒绝当“调包侠”。
- **手写线性回归与反向传播**：基于 NumPy 实现梯度下降。
- **手写 Transformer**：从零实现 Multi-Head Attention、位置编码、Mask 机制。
- **复现 Mini GPT**：在莎士比亚数据集上训练字符级语言模型，理解自回归生成机制。

### 2. 大模型微调与对齐 (`day27` - `day29`)
基于 HuggingFace 生态和 `trl` 库，完成完整的微调闭环。
- **SFT（监督微调）**：构建指令微调数据，跑通 Qwen 模型的全参微调。
- **LoRA / QLoRA**：手写 LoRA 层（低秩分解），实现极低显存的参数高效微调。
- **DPO（直接偏好优化）**：推导 DPO 数学公式，训练 Reward Model，完成人类偏好对齐。

### 3. 推理加速与 Infra 源码剖析 (`day30`, `day36` - `day39`, `vllm-source-study`)
深入大模型工程底层，解决显存与速度瓶颈。
- **混合精度与显存优化**：实践 AMP（自动混合精度）、梯度累积、梯度检查点。
- **HuggingFace Accelerate**：实现单套代码适配单卡/多卡分布式训练。
- **vLLM 源码拆解（v0.5.4）**：深度剖析 `Scheduler` 和 `BlockManager`，理解 **PagedAttention** 显存分页管理、**Continuous Batching** 动态批处理、以及 **Prefix Caching** 机制。

### 4. 算法项目：投资量化模型 (`investment_project`)
- 结合 FinBERT 金融新闻情感分析与 Transformer，构建股票收益率预测系统。
- 严格时间切分、数据归一化、回测（计算 IC、夏普比率、最大回撤）。

### 5. AI Agent 全栈开发 (`kisaragi_nya_bot`, `perplexity-clone`)
将大模型真正落地为生产力工具和娱乐应用。

#### 🔍 本地版 Perplexity (Agentic RAG 搜索引擎)
- **智能路由 (Router)**：判断闲聊、事实查询与深度检索。
- **多跳搜索 (Multi-hop)**：自动拆解复杂查询，结合 BGE-Reranker 实现精准重排与自反思补搜。
- **流式体验**：基于 Gradio 实现带打字机效果的 Web UI。

#### 🤖 Kisaragi_Nya 赛博水群 AI (QQ 机器人)
基于 `NoneBot2` + `NapCat` + `Ollama (Qwen2.5-14B)` 构建的本地拟人化智能体。
- **拟人化交互**：多段延迟发送模拟真人打字，支持上下文记忆（L1）。
- **长期事实库 (L3 Memory)**：引入 ChromaDB 向量检索，跨时空“清算”群友吹过的牛逼。
- **动态情感状态机**：根据聊天内容动态加减“好感度”，记录群友画像，实现定向毒舌与破防自闭。
- **多模态视觉**：接入 Qwen-VL 模型，看图精准嘲讽。
- **工业级高并发与风控**：基于 `asyncio` 实现非阻塞并发处理；包含全局限流、敏感词熔断、心跳定时持久化（防止高频 IO 崩溃）。

---

## 🛠️ 技术栈一览

- **语言与框架**: Python 3.10+, PyTorch, HuggingFace (Transformers, PEFT, TRL, Accelerate)
- **大模型与推理**: Ollama, vLLM, Qwen2.5, BGE (Embedding/Reranker)
- **Agent 与 RAG**: NoneBot2, Gradio, ChromaDB, DuckDuckGo Search, Trafilatura
- **工程化工具**: Git, FastAPI, asyncio, bitsandbytes

---

## 📂 核心项目目录结构

```text
ai-learning/
├── day01 - day25/               # 基础算法、Transformer 与 Mini GPT
├── day27 - day29/               # SFT, LoRA, DPO 微调实战
├── day30 - day40/               # vLLM 入门, 混合精度, 分布式训练
├── vllm-source-study/           # vLLM 底层源码剖析 (PagedAttention)
├── perplexity-clone/            # 本地化 Agentic RAG 搜索引擎
└── kisaragi_nya_bot/            # 基于 NoneBot2 的赛博群友 QQ 机器人
```

---

## 🚀 如何运行

### 环境配置
```bash
git clone https://github.com/你的用户名/ai-learning.git
cd ai-learning
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 运行赛博群友 (需提前安装 Ollama 和 NapCat)
1. 拉取本地模型：`ollama run qwen2.5:14b` 和 `ollama run qwen2.5-vl:7b`
2. 进入机器人目录：`cd kisaragi_nya_bot`
3. 配置 `.env` 文件（填入你的 Ollama 地址和机器人配置）。
4. 启动：`nb run`

---

## 💡 学习感悟与总结

1. **基础决定上限**：手写过 Attention 的 `Q @ K.T` 和反向传播后，再看大模型源码不再是黑盒。
2. **数据与工程同等重要**：Prompt 调得再好，也不如 Reranker 选得准；算法再牛，也要靠 `asyncio` 防阻塞。
3. **落地是检验真理的唯一标准**：从跑通脚本到做成能抗住高并发、防封号、有记忆的 QQ 机器人，中间填平了无数工程坑。

> **持续更新中...**