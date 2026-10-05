import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B"
print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"模型参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
print(f"模型配置：")
print(f"  隐藏维度 d_model: {model.config.hidden_size}")
print(f"  层数 num_layers: {model.config.num_hidden_layers}")
print(f"  注意力头数 num_heads: {model.config.num_attention_heads}")
print(f"  词表大小 vocab_size: {model.config.vocab_size}")

text = "人工智能是研究如何让计算机完成需要人类智能的任务的学科。"

print(f"\n原始文本：{text}")
ids = tokenizer(text, return_tensors="pt")
print(f"Token IDs：{ids['input_ids'][0][:10]}...")
print(f"Token 数量：{len(ids['input_ids'][0])}")
decoded = tokenizer.decode(ids["input_ids"][0], skip_special_tokens=True)
print(f"解码结果：{decoded}")

prompt = "人工智能是"
inputs = tokenizer(prompt, return_tensors="pt")
print(f"\n输入：{prompt}")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens = 50,
        temperature = 0.7,
        do_sample = True,
        top_p = 0.9,
    )
generated = tokenizer.decode(outputs[0], skip_special_tokens = True)
print(f"生成：{generated}")

print(f"\n模型结构（前 3 层）：")
for i, (name, module) in enumerate(model.named_children()):
    if i >= 3:
        break
    print(f"  {name}: {type(module).__name__}")