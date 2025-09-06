import torch
import torch.nn as nn
from embedding_layer import EmbeddingLayer
from mlp_layer import MultiLayerPerceptron
from gate_layer import Gate


class PLEModel(nn.Module):
    """
    PLE 主体模型 (简化版)
    """
    def __init__(self, categorical_field_dims, numerical_num, embed_dim,
                 bottom_mlp_dims, tower_mlp_dims, task_num,
                 shared_expert_num, specific_expert_num, dropout):
        super().__init__()

        # Embedding 层
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim

        self.task_num = task_num
        self.layers_num = len(bottom_mlp_dims)

        # 定义专家和门控
        self.task_experts = nn.ModuleList()
        self.task_gates = nn.ModuleList()
        self.share_experts = nn.ModuleList()
        self.share_gates = nn.ModuleList()

        for i in range(self.layers_num):
            input_dim = self.embed_output_dim if i == 0 else bottom_mlp_dims[i-1]

            # 共享专家
            share_expert_list = nn.ModuleList([
                MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False)
                for _ in range(shared_expert_num)
            ])
            self.share_experts.append(share_expert_list)
            self.share_gates.append(Gate(input_dim, shared_expert_num + task_num * specific_expert_num))

            # 任务专家和门控
            task_expert_list = []
            task_gate_list = []
            for _ in range(task_num):
                experts = nn.ModuleList([
                    MultiLayerPerceptron(input_dim, [bottom_mlp_dims[i]], dropout, output_layer=False)
                    for _ in range(specific_expert_num)
                ])
                gate = Gate(input_dim, shared_expert_num + specific_expert_num)
                task_expert_list.append(experts)
                task_gate_list.append(gate)
            self.task_experts.append(nn.ModuleList(task_expert_list))
            self.task_gates.append(nn.ModuleList(task_gate_list))

        # Task Tower
        self.towers = nn.ModuleList([
            MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout)
            for _ in range(task_num)
        ])

    def forward(self, categorical_x, numerical_x):
        """
        categorical_x: LongTensor, (batch_size, num_categorical_fields)
        numerical_x: FloatTensor, (batch_size, numerical_num)
        """
        categorical_emb = self.embedding(categorical_x)  # (batch, fields, dim)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = torch.cat([categorical_emb, numerical_emb], dim=1).view(-1, self.embed_output_dim)

        task_fea = [emb for _ in range(self.task_num + 1)]  # 每个任务 + 共享
        for i in range(self.layers_num):
            share_output = [expert(task_fea[-1]).unsqueeze(1) for expert in self.share_experts[i]]
            task_output_list = []

            # 每个任务
            for j in range(self.task_num):
                task_output = [expert(task_fea[j]).unsqueeze(1) for expert in self.task_experts[i][j]]
                task_output_list.extend(task_output)
                mix_output = torch.cat(task_output + share_output, dim=1)
                gate_value = self.task_gates[i][j](task_fea[j]).unsqueeze(1)
                task_fea[j] = torch.bmm(gate_value, mix_output).squeeze(1)

            # 共享门（最后一层不需要）
            if i != self.layers_num - 1:
                mix_output = torch.cat(task_output_list + share_output, dim=1)
                gate_value = self.share_gates[i](task_fea[-1]).unsqueeze(1)
                task_fea[-1] = torch.bmm(gate_value, mix_output).squeeze(1)

        results = [torch.sigmoid(self.towers[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results
