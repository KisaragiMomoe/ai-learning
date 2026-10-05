import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

data = [
    {"instruction": "什么是人工智能？", "output": "人工智能是研究如何让计算机模拟人类智能的学科，包括学习、推理、感知和决策。"},
    {"instruction": "什么是机器学习？", "output": "机器学习是人工智能的一个分支，让计算机从数据中自动学习规律，而不需要人工编写规则。"},
    {"instruction": "什么是深度学习？", "output": "深度学习是机器学习的一个分支，使用多层神经网络来提取数据的特征。"},
    {"instruction": "什么是神经网络？", "output": "神经网络是由很多神经元组成的计算模型，每个神经元接收输入，进行计算，然后输出。"},
    {"instruction": "什么是Transformer？", "output": "Transformer是一种基于注意力机制的神经网络架构，是现代大语言模型的基础。"},
    {"instruction": "什么是大语言模型？", "output": "大语言模型是用海量文本数据训练的深度学习模型，能生成文本、回答问题、翻译语言。"},
    {"instruction": "什么是注意力机制？", "output": "注意力机制让模型在处理序列时，能够关注输入中的重要部分，而不是一视同仁。"},
    {"instruction": "什么是反向传播？", "output": "反向传播是一种训练神经网络的算法，通过链式法则计算损失对每个参数的梯度。"},
    {"instruction": "什么是梯度下降？", "output": "梯度下降是一种优化算法，沿着梯度的反方向更新参数，让损失函数变小。"},
    {"instruction": "什么是过拟合？", "output": "过拟合是模型在训练数据上表现很好，但在新数据上表现很差的现象。"},
    {"instruction": "什么是正则化？", "output": "正则化是一种防止过拟合的技术，通过在损失函数中加入惩罚项，限制模型复杂度。"},
    {"instruction": "什么是Dropout？", "output": "Dropout是一种正则化方法，训练时随机丢弃一部分神经元，防止模型过度依赖某些神经元。"},
    {"instruction": "什么是交叉熵？", "output": "交叉熵是衡量两个概率分布差异的指标，常用于分类任务的损失函数。"},
    {"instruction": "什么是最大似然估计？", "output": "最大似然估计是找到一组参数，使得观测数据出现的概率最大。"},
    {"instruction": "什么是SFT？", "output": "SFT是监督微调，用问题-答案数据训练模型，让模型学会回答问题。"},
    {"instruction": "什么是LoRA？", "output": "LoRA是一种参数高效微调方法，不修改原模型权重，只训练一小部分低秩矩阵。"},
    {"instruction": "什么是DPO？", "output": "DPO是直接偏好优化，用偏好数据对齐模型，不需要训练奖励模型。"},
    {"instruction": "什么是KV Cache？", "output": "KV Cache是推理时缓存已计算的Key和Value，避免重复计算，加快生成速度。"},
    {"instruction": "什么是量化？", "output": "量化是用更少的位数存储模型参数，减少显存占用，可能损失一点精度。"},
    {"instruction": "什么是困惑度？", "output": "困惑度是交叉熵的指数，衡量模型预测下一个词时的不确定性。"},
]

print(f"数据条数：{len(data)}")

model_name = "Qwen/Qwen2.5-0.5B"
print(f"加载 Tokenizer 和模型：{model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"模型参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

def format_data(example):
    """把 instruction/output 拼成对话模板"""
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize = False)
    return {"text": text}
dataset = Dataset.from_list(data)
dataset = dataset.map(format_data)
print(f"\n格式化后的第一条数据：")
print(dataset[0]["text"][:200], "...")

def tokenize(example):
    result = tokenizer(
        example["text"],
        truncation = True,
        max_length = 256,
        padding = "max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize, remove_columns = dataset.column_names)
print(f"\nTokenize 后的数据：")
print(f"  input_ids 形状：{len(tokenized_dataset[0]['input_ids'])}")
print(f"  样本数：{len(tokenized_dataset)}")
training_args = TrainingArguments(
    output_dir="./day27_sft_output",
    num_train_epochs = 10,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-5,
    warmup_steps = 10,
    logging_steps = 5,
    save_steps = 50,
    save_total_limit = 5,
    fp16 = False,
    bf16 = False,
    report_to = "none",
    remove_unused_columns = False,
)

trainer = Trainer(
    model = model, 
    args = training_args,
    train_dataset = tokenized_dataset,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("\n开始训练...")
trainer.train()
save_path = "./day27_sft_model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

print(f"\n模型已保存到：{save_path}")
print("\n" + "="*50)
print("测试微调后的模型：")
print("="*50)

def ask(question):
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

test_questions = [
    "什么是人工智能？",
    "什么是LoRA？",
    "什么是Transformer？",
    "今天天气怎么样？",   # 这个问题不在训练数据里，看模型怎么回答
]

for q in test_questions:
    print(f"\n问：{q}")
    print(f"答：{ask(q)}")
