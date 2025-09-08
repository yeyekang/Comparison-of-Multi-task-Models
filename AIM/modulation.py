import torch
import torch.nn as nn
import torch.nn.functional as F

class AIMModulation(nn.Module):
    def __init__(self, hidden_dim, mode="gating", num_heads=4, dropout=0.1, temperature=1.0, lambda_res=1.0):
        """
        AIM 调制模块
        :param hidden_dim: 表示向量维度 (h_t 的维度)
        :param mode: 调制方式 ["gating", "attention", "residual"]
        :param num_heads: 注意力头数 (仅 attention 模式使用)
        :param dropout: dropout 比例
        :param temperature: softmax 温度参数 (仅 attention 模式使用)
        :param lambda_res: 残差修正项缩放系数 (仅 residual 模式使用)
        """
        super(AIMModulation, self).__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.temperature = temperature
        self.lambda_res = lambda_res

        if mode == "gating":
            self.gate_layer = nn.Linear(2 * hidden_dim, hidden_dim)

        elif mode == "attention":
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        elif mode == "residual":
            self.proj = nn.Linear(hidden_dim, hidden_dim)
            self.act = nn.ReLU()

    def forward(self, h_t, H_other):
        """
        :param h_t: 当前任务表示, shape = [batch, hidden_dim]
        :param H_other: 其他任务表示, shape = [batch, num_tasks-1, hidden_dim]
        :return: 融合后的表示, shape = [batch, hidden_dim]
        """
        if self.mode == "gating":
            # 拼接 [h_t, mean(H_other)]
            h_other = H_other.mean(dim=1)   # [batch, hidden_dim]
            concat = torch.cat([h_t, h_other], dim=-1)  # [batch, 2*hidden_dim]
            g = torch.sigmoid(self.gate_layer(concat))  # [batch, hidden_dim]
            h_t_new = h_t + g * h_other

        elif self.mode == "attention":
            # 输入到多头注意力: query=h_t, key=value=H_other
            # 需要 reshape -> [batch, seq_len, dim]
            query = h_t.unsqueeze(1)          # [batch, 1, hidden_dim]
            key = H_other                     # [batch, num_tasks-1, hidden_dim]
            value = H_other

            attn_output, _ = self.attn(query, key, value)  # [batch, 1, hidden_dim]
            h_t_new = h_t + attn_output.squeeze(1)

        elif self.mode == "residual":
            h_other = H_other.mean(dim=1)  # [batch, hidden_dim]
            delta = self.act(self.proj(h_other))  # [batch, hidden_dim]
            h_t_new = h_t + self.lambda_res * delta

        else:
            raise ValueError(f"Unknown mode {self.mode}")

        return h_t_new
