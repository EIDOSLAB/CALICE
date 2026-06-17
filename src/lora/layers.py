import torch
from custom_comp.layers.win_attention import WindowAttention
import torch.nn as nn

def swish(x, beta=1.0):
    return x * torch.sigmoid(x * beta)


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, activation = nn.Identity ):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.activation = activation

    def forward(self, x):
        x = x @ self.W_a
        x = self.activation(x)
        x = x @ self.W_b
        return self.alpha * x
        # x = self.alpha * (x @ self.W_a @ self.W_b)
        # return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, activation = nn.Identity, divide_rank = False):
        super().__init__()
        self.linear = linear

        if divide_rank:
            rank = linear.in_features // rank
            print(f'rank -> {rank}')
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, activation
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
    

class LoraWithLora(torch.nn.Module):
    def __init__(self, linear_with_lora:LinearWithLoRA, rank, beta, activation = nn.Identity, divide_rank = False):
        super().__init__()

        self.linear_with_lora = linear_with_lora
        if divide_rank:
            rank = linear_with_lora.linear.in_features // rank
            print(f'rank -> {rank}')

        self.lora = LoRALayer(
            linear_with_lora.linear.in_features, linear_with_lora.linear.out_features, rank, beta, activation
        )
    
    def forward(self, x):
        return self.linear_with_lora(x) + self.lora(x)


# class AttentionWithLoRA(WindowAttention):
#     def __init__(self, dim=192, window_size=..., num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0):
#         super().__init__(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

#         self.lora


if __name__ == '__main__':
    l = LoRALayer(in_dim=198, out_dim=198, rank=8, alpha=0.5, activation=swish).cuda()
    x = torch.rand((8,100,198)).cuda()
    print(l(x).shape)