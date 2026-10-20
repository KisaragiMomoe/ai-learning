import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

data = [
    {"prompt": "什么是人工智能？", 
     "chosen": "人工智能是研究如何让计算机模拟人类智能的学科，包括学习、推理和决策。", 
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
    # 真实场景下，你需要至少几千条这样的数据
]

dataset = Dataset.from_list(data)
print(f"数据条数：{len(dataset)}")

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels = 1,
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
)

# def preprocess_function(examples):
#     chosen_texts = [p + a for p, a in zip(examples["prompt"], examples["chosen"])]
#     rejected_texts = [p + a for p, a in zip(examples["prompt"], examples["rejected"])]
#     chosen_tokens = tokenizer(chosen_texts, truncation = True, max_length = 256, padding = "max_length")
#     rejected_tokens = tokenizer(rejected_texts, truncation = True, max_length = 256, padding = "max_length")
#     return {
#         "input_ids_chosen": chosen_tokens["input_ids"],
#         "attention_mask_chosen": chosen_tokens["attention_mask"],
#         "input_ids_rejected": rejected_tokens["input_ids"],
#         "attention_mask_rejected": rejected_tokens["attention_mask"],
#     }

# tokenized_dataset = dataset.map(
#     preprocess_function,
#     batched = True,
#     remove_columns = dataset.column_names
# )

training_args = RewardConfig(
    output_dir="./day42_reward_model",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    learning_rate=1e-4,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    bf16=torch.cuda.is_bf16_supported(),
)

trainer = RewardTrainer(
    model = model,
    args = training_args,
    train_dataset = dataset,
    processing_class = tokenizer,
)

print("\n开始训练 Reward Model...")
trainer.train()

print("\n测试 Reward Model 打分：")
model.eval()

def get_score(text):
    inputs = tokenizer(text, return_tensors = "pt", truncation = True).to(model.device)
    with torch.no_grad():
        score = model(**inputs).logits[0][0].item()
    return score

test_prompt = "什么是人工智能？"
good_answer = test_prompt + "人工智能是研究如何让计算机模拟人类智能的学科。"
bad_answer = test_prompt + "人工智能就是机器人。"

good_score = get_score(good_answer)
bad_score = get_score(bad_answer)

print(f"好回答得分：{good_score:.4f}")
print(f"坏回答得分：{bad_score:.4f}")
print(f"差值：{good_score - bad_score:.4f}（差值越大，说明模型越能分辨好坏）")
