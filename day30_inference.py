import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

print("\n" + "="*50)
print("实验 1：KV Cache 对速度的影响")
print("="*50)

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True).to(device)
model.eval()

prompt = "人工智能是"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

start = time.time()
with torch.no_grad():
    outputs_with_cache = model.generate(
        **inputs,
        max_new_tokens = 100,
        do_sample = False,
        use_cache = True,
    )
time_with_cache = time.time() - start

start = time.time()
with torch.no_grad():
    outputs_without_cache = model.generate(
        **inputs,
        max_new_tokens = 100,
        do_sample = False,
        use_cache = False,
    )
time_without_cache = time.time() - start

print(f"开启 KV Cache：{time_with_cache:.2f} 秒")
print(f"关闭 KV Cache：{time_without_cache:.2f} 秒")
print(f"加速比：{time_without_cache / time_with_cache:.2f}x")

print("\n" + "="*50)
print("实验 2：量化对显存和速度的影响")
print("="*50)

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None
bnb_config_8bit = BitsAndBytesConfig(load_in_8bit = True)
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config_8bit,
    device_map = "auto" if torch.cuda.is_available() else None,
)

def get_model_size(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / 1e6
print(f"8bit 模型大小：{get_model_size(model_8bit):.1f} MB")

bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
)
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = bnb_config_4bit,
    device_map = "auto" if torch.cuda.is_available() else None,
)

print(f"4bit 模型大小：{get_model_size(model_4bit):.1f} MB")

print("\n" + "="*50)
print("实验 3：4bit 量化模型生成")
print("="*50)

inputs = tokenizer(prompt, return_tensors = "pt").to(device)
with torch.no_grad():
    outputs = model_4bit.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )

print(f"输入：{prompt}")
print(f"4bit 生成：{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
print("\n" + "="*50)

print("实验 4：加载 Day 28 的 LoRA 权重")
print("="*50)

import os
lora_path = "./day28_lora_model"

if (os.path.exists(lora_path)):
    base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True)
    model_with_lora = PeftModel.from_pretrained(base_model, lora_path)
    model_with_lora = model_with_lora.to(device)
    model_with_lora.eval()
    def ask(model, question):
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
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
            )

        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:].cpu(), skip_special_tokens=True)

    for q in ["什么是人工智能？", "什么是LoRA？"]:
        print(f"\n问：{q}")
        print(f"答：{ask(model_with_lora, q)}")
else:
    print(f"未找到 {lora_path}，跳过。请先运行 Day 28。")

print("\n" + "="*50)
print("实验 5：合并 LoRA 权重")
print("="*50)

if os.path.exists(lora_path):
    merged_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True),
        lora_path,
    )
    merged_model = merged_model.merge_and_unload()
    print("LoRA 权重已合并到原模型")
    merged_path = "./day30_merged_model"
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"合并后的模型已保存到：{merged_path}")
    print("合并后推理时不需要再加载 LoRA，速度更快。")