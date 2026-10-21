import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank = 8, alpha = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha

        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False

        self.lora_A = nn.Linear(in_features, rank, bias = False)
        self.lora_B = nn.Linear(rank, out_features, bias = False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a = math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.scaling = self.alpha / self.rank

    def forward(self, x):
        original_output = self.linear(x)
        lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return original_output + lora_output

if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 10, 512
    x = torch.randn(batch_size, seq_len, d_model)

    original_linear = nn.Linear(d_model, d_model)
    out_original = original_linear(x)

    lora_linear = LoRALayer(d_model, d_model, rank = 8, alpha = 16)

    lora_linear.linear.weight.data = original_linear.weight.data.clone()
    lora_linear.linear.bias.data = original_linear.bias.clone()

    out_lora = lora_linear(x)

    print(f"原始输出形状: {out_original.shape}")
    print(f"LoRA 输出形状: {out_lora.shape}")
    print(f"初始状态下两者误差: {(out_original - out_lora).abs().max().item():.6f} (应该接近 0)")

    # 检查可训练参数
    trainable_params = sum(p.numel() for p in lora_linear.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_linear.parameters())
    print(f"可训练参数: {trainable_params}")
    print(f"总参数: {total_params}")
    print(f"训练参数占比: {100 * trainable_params / total_params:.2f}%")
