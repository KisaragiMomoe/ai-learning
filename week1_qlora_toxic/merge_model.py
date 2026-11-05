import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_id = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = "./qwen-toxic-lora-final"
save_path = "./qwen-toxic-merged"

print("加载基础模型...")

base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype = torch.float16,
    device_map = "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("挂载 LoRA 并进行合并...")
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("合并完成！文件在：", save_path)
