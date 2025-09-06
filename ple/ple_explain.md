1.简介

精髓在多个CGC层叠加。 CGC层包含共享专家和任务专属专家，输入包括任务向量和共享向量。

这样保证更能解决负迁移问题。

2.训练经验（来源github）

参考实现的默认/常见设定（可作为起点）

层数（num_levels / num_cgc_layers）：2（论文/实现中常见）；可尝试 1–3。

每个专家（expert）内部：一个小 DNN，例如 [256,128] hidden units（DeepCTR 默认示例）。

共享专家数量 S：1–4（根据模型规模），任务专属专家 P：1–4。MTReclib 中 expert_num 默认是 8（所有专家之和或单组数可配置）。batch-size 建议大（如 2048／1024），lr 初始 1e-3。

Gate 网络：通常是 1 层小 DNN（64 units）或直接线性变换 + softmax。

正则化：Dropout、L2、早停；若专家过多会有过拟合或“专家塌陷”（expert collapse）问题，可使用 expert normalization / sparsity penalties（后续工作中常见改进）。
