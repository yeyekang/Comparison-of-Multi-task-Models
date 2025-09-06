import torch
import torch.nn as nn

class Gate(nn.Module):
    """
    门控层: 输入 hidden，输出对专家的 softmax 权重
    """
    def __init__(self, input_dim, expert_num):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, expert_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        # 输出: (batch_size, expert_num)
        return self.gate(x)
