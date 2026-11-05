import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./qwen-toxic-lora-final"

tokenizer = AutoTokenizer.from_pretrained(model_id)
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

test_prompts = ["兄弟们我科目二挂了。", "今天被老板开除了，好难过。"]

for prompt in test_prompts:
    # 👈 核心修复：注入毒舌 System Prompt
    messages = [
        {"role": "system", "content": "你是一个二次元毒舌群友，说话极其刻薄，喜欢用网络烂梗，口头禅是'喵'。请用极短的一句话嘲讽对方。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7, do_sample=True)
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\n用户: {prompt}")
    print(f"赛博小二: {response}")