# Qwen2.5 毒舌人格微调实战（QLoRA + DPO）

本项目记录了如何通过高质量数据构建、SFT 监督微调与 DPO 偏好对齐，将一个通用的 Qwen 模型，定制化为具备特定“二次元毒舌”人格的垂直 Agent。

## 🛠️ 技术栈与核心难点
- **模型**：Qwen2.5-7B-Instruct (4-bit QLoRA 量化加载)
- **训练框架**：HuggingFace `transformers`, `peft`, `trl`
- **显存占用**：约 8 GB（RTX 5060 可流畅运行）
- **核心突破**：
  1. **拒绝采样数据合成**：利用 LLM-as-a-Judge 生成高质量、高毒的二次元对话数据。
  2. **Loss Mask 技术**：重写了 `Dataset` 与 `DataCollator`，将用户提问部分的 Labels 设置为 `-100`，使模型 100% 专注学习回答风格。
  3. **QLoRA 量化**：使用 NF4 量化（`bnb_4bit_quant_type="nf4"`）极大降低微调门槛，并配合 `paged_adamw_8bit` 优化器防止显存溢出。
  4. **DPO 偏好对齐**：利用 `DPOTrainer`（隐藏 Reference 模型），通过数学推导直接拉开“毒舌回答”与“废话回答”的概率差。

## 🚀 实验对比
1. **Base Model (原始模型)**：典型的 AI 助手风格，遇到问题输出长篇大论的安慰与建议。
2. **SFT Model (监督微调)**：学会了特定的口癖（喵、w），但在某些语境下仍会退回温柔风格。
3. **DPO Model (偏好对齐)**：极大地压制了“温柔废话”的生成概率，全面覆盖毒舌、傲娇、阴阳怪气的语气。

## 📂 目录结构
```text
week1_qlora_toxic/
├── data/
│   └── train.jsonl          # SFT 指令微调数据
│   └── dpo_data.jsonl       # DPO 偏好对数据
├── train_sft.py             # 带 Loss Mask 的 QLoRA 训练脚本
├── train_dpo.py             # DPO 训练脚本
├── merge_model.py           # 合并 LoRA 权重并导出
└── test_model.py            # 测试脚本（注入 System Prompt）
```


## 📝 踩坑记录（DevLog）
- 遇到 `huggingface_hub` 与 `datasets` 的版本冲突问题，通过 `pip install "datasets<5.1"` 降级修复。
- 发现 SFT 后模型仍不毒舌，排查定位为未对 User Prompt 添加 Loss Mask，导致模型算力被问题本身消耗。