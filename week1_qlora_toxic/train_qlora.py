import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

model_id = "Qwen/Qwen2.5-7B-Instruct" 
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config = bnb_config,
    device_map = "auto"
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r = 16,
    lora_alpha = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def format_dataset(example):
    # 1. 拿到用户输入和 AI 回答的文本
    user_text = tokenizer.apply_chat_template([example["messages"][0]], tokenize=False, add_generation_prompt=True)
    assistant_text = example["messages"][1]["content"] + tokenizer.eos_token
    
    # 2. 分别 Tokenize
    user_ids = tokenizer(user_text, add_special_tokens=False).input_ids
    assistant_ids = tokenizer(assistant_text, add_special_tokens=False).input_ids
    
    # 3. 拼接
    input_ids = user_ids + assistant_ids
    
    # 4. 核心：把 User 部分对应的 labels 设为 -100（不计算 Loss），只学回答部分
    labels = [-100] * len(user_ids) + assistant_ids
    
    # 5. 截断到最大长度
    max_len = 256
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
        
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }

dataset = load_dataset("json", data_files="train.jsonl", split="train")
tokenized_dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)

training_args = TrainingArguments(
    output_dir = "./qwen-toxic-lora",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    learning_rate = 2e-4,
    num_train_epochs = 5,
    logging_steps = 2,
    bf16 = True,
    save_strategy = "epoch",
    optim = "paged_adamw_8bit"
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_dataset,
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding = True, label_pad_token_id = -100)   
)

print("开始 QLoRA 微调...")
trainer.train()

# 6. 保存 LoRA 权重
trainer.model.save_pretrained("./qwen-toxic-lora-final")
tokenizer.save_pretrained("./qwen-toxic-lora-final")
print("微调完成！")

