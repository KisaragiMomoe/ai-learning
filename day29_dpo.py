import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
data = [
    {"prompt": "什么是人工智能？",
     "chosen": "人工智能是研究如何让计算机模拟人类智能的学科，包括学习、推理、感知和决策。",
     "rejected": "人工智能就是机器人。"},
    {"prompt": "什么是机器学习？",
     "chosen": "机器学习是人工智能的一个分支，让计算机从数据中自动学习规律，而不需要人工编写规则。",
     "rejected": "机器学习就是让机器自己学习。"},
    {"prompt": "什么是深度学习？",
     "chosen": "深度学习是机器学习的一个分支，使用多层神经网络来提取数据的特征。",
     "rejected": "深度学习就是很深的网络。"},
    {"prompt": "什么是Transformer？",
     "chosen": "Transformer是一种基于注意力机制的神经网络架构，是现代大语言模型的基础。",
     "rejected": "Transformer就是变形金刚。"},
    {"prompt": "什么是大语言模型？",
     "chosen": "大语言模型是用海量文本数据训练的深度学习模型，能生成文本、回答问题、翻译语言。",
     "rejected": "大语言模型就是很大的模型。"},
    {"prompt": "什么是注意力机制？",
     "chosen": "注意力机制让模型在处理序列时，能够关注输入中的重要部分，而不是一视同仁。",
     "rejected": "注意力机制就是让模型集中注意力。"},
    {"prompt": "什么是过拟合？",
     "chosen": "过拟合是模型在训练数据上表现很好，但在新数据上表现很差的现象。",
     "rejected": "过拟合就是拟合得太好了。"},
    {"prompt": "什么是正则化？",
     "chosen": "正则化是一种防止过拟合的技术，通过在损失函数中加入惩罚项，限制模型复杂度。",
     "rejected": "正则化就是让模型变正规。"},
    {"prompt": "什么是SFT？",
     "chosen": "SFT是监督微调，用问题-答案数据训练模型，让模型学会回答问题。",
     "rejected": "SFT就是微调。"},
    {"prompt": "什么是LoRA？",
     "chosen": "LoRA是一种参数高效微调方法，不修改原模型权重，只训练一小部分低秩矩阵。",
     "rejected": "LoRA就是低秩适应。"},
    {"prompt": "什么是KV Cache？",
     "chosen": "KV Cache是推理时缓存已计算的Key和Value，避免重复计算，加快生成速度。",
     "rejected": "KV Cache就是缓存。"},
    {"prompt": "什么是量化？",
     "chosen": "量化是用更少的位数存储模型参数，减少显存占用，可能损失一点精度。",
     "rejected": "量化就是数字化。"},
    {"prompt": "什么是困惑度？",
     "chosen": "困惑度是交叉熵的指数，衡量模型预测下一个词时的不确定性。",
     "rejected": "困惑度就是模型有多困惑。"},
    {"prompt": "什么是梯度下降？",
     "chosen": "梯度下降是一种优化算法，沿着梯度的反方向更新参数，让损失函数变小。",
     "rejected": "梯度下降就是往下走。"},
    {"prompt": "什么是反向传播？",
     "chosen": "反向传播是一种训练神经网络的算法，通过链式法则计算损失对每个参数的梯度。",
     "rejected": "反向传播就是往回传。"},
]
dataset = Dataset.from_list(data)
print(f"偏好数据条数：{len(dataset)}")

sft_model_path = "./day27_sft_model"

tokenizer = AutoTokenizer.from_pretrained(sft_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True)
ref_model = AutoModelForCausalLM.from_pretrained(sft_model_path, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
ref_model = ref_model.to(device)
ref_model.eval()
print(f"使用设备：{device}")

lora_config = LoraConfig(
    r = 32,
    lora_alpha = 64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 加更多层
    lora_dropout = 0.05,
    bias = "none",
    task_type = TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数：{trainable:,} / 总参数：{total:,} ({100 * trainable / total:.2f}%)")
print_trainable_parameters(model)

training_args = DPOConfig(
    output_dir = "./day29_dpo_output",
    num_train_epochs = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    learning_rate = 1e-6,
    beta = 0.1,
    logging_steps = 5,
    save_steps = 50,
    save_total_limit = 2,
    fp16 = False,
    bf16 = False,
    report_to = "none",
    remove_unused_columns = False, 
)

trainer = DPOTrainer(
    model = model,
    ref_model = ref_model,
    args = training_args,
    train_dataset = dataset,
    processing_class = tokenizer,
)

print("\n开始 DPO 训练...")
trainer.train()

save_path = "./day29_dpo_model"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nDPO 模型已保存到：{save_path}")

def ask(question):
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer(text, return_tensors = "pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 80,
            temperature = 0.7,
            do_sample = True,
            top_p = 0.9,
            repetition_penalty = 1.2,
            no_repeat_ngram_size = 3,
            eos_token_id = tokenizer.eos_token_id,
        )    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:].cpu(), skip_special_tokens=True)
    return response
print("\n" + "="*50)
print("测试 DPO 后的模型：")
print("="*50)

for q in ["什么是人工智能？", "什么是LoRA？", "什么是Transformer？"]:
    print(f"\n问：{q}")
    print(f"答：{ask(q)}")