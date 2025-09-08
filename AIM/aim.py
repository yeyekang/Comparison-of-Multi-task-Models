# aim.py
import torch
import torch.nn as nn
from modulation import ModulationLayer


class AIMModel(nn.Module):
    """
    Adaptive Information Modulation (AIM) 模型
    """

    def __init__(self, input_dim, hidden_dim, task_num, modulation_type="gating"):
        super(AIMModel, self).__init__()
        self.shared_bottom = nn.Linear(input_dim, hidden_dim)  # 共享层
        self.task_embeddings = nn.Parameter(torch.randn(task_num, hidden_dim))  # 每个任务一个embedding
        self.modulation_layers = nn.ModuleList(
            [ModulationLayer(hidden_dim, modulation_type) for _ in range(task_num)]
        )
        self.task_towers = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(task_num)]
        )

    def forward(self, x):
        """
        x: [batch, input_dim]
        输出: [task_num][batch, 1]
        """
        shared_feat = torch.relu(self.shared_bottom(x))  # 共享特征
        outputs = []

        for i, (mod_layer, tower) in enumerate(zip(self.modulation_layers, self.task_towers)):
            task_emb = self.task_embeddings[i].unsqueeze(0).expand(x.size(0), -1)
            modulated_feat = mod_layer(shared_feat, task_emb)  # 调制
            out = torch.sigmoid(tower(modulated_feat))  # 每个任务输出
            outputs.append(out)

        return outputs


if __name__ == "__main__":
    # 模拟输入测试
    batch_size, input_dim, hidden_dim, task_num = 4, 16, 8, 3
    x = torch.randn(batch_size, input_dim)

    model = AIMModel(input_dim, hidden_dim, task_num, modulation_type="gating")
    y = model(x)

    for i, out in enumerate(y):
        print(f"任务 {i} 输出: {out.detach().cpu().numpy()}")
