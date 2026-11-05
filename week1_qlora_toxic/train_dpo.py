import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

model_id = "Qwen/Qwen2.5-7B-Instruct"
sft_lora_path = "./qwen-toxic-lora-final" # 刚刚 SFT 训出来的补丁
dpo_output_path = "./qwen-toxic-dpo"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

print("加载基础模型并挂载 SFT LoRA...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config = bnb_config, 
    device_map = "auto"
)

model = PeftModel.from_pretrained(base_model, sft_lora_path)

training_args = DPOConfig(
    output_dir = dpo_output_path,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-5,
    beta = 0.1,
    num_train_epochs = 3,
    logging_steps = 1,
    bf16 = True,
    optim = "paged_adamw_8bit",
    remove_unused_columns = False,
)

dataset = load_dataset("json", data_files="dpo_data.jsonl", split="train")

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = training_args,
    train_dataset = dataset,
    processing_class = tokenizer
)

print("开始 DPO 训练...")
dpo_trainer.train()

print("DPO 训练完成，保存权重...")
dpo_trainer.model.save_pretrained(dpo_output_path)
tokenizer.save_pretrained(dpo_output_path)
print("大功告成！")
