import dgl.sparse as dglsp
import torch
import torch.nn as nn


class GraphWaveletLayer(nn.Module):
    """
    单层图小波神经网络层。

    在波浪域应用可学习的对角滤波器以执行局部谱过滤操作。

    :param input_dim: 输入节点特征维度。
    :param output_dim: 输出节点特征维度。
    :param wavelet_shape: 小波矩阵形状 (num_nodes, num_nodes)。
    :param dropout_rate: 应用于输出的dropout率。
    :param device: 计算设备（例如 'cpu' 或 'cuda'）。
    """

    def __init__(self, input_dim, output_dim, wavelet_shape, dropout_rate, device):
        super(GraphWaveletLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes, self.num_basis = wavelet_shape  # 波浪形形状: (N, N)
        self.device = device
        self.dropout_rate = dropout_rate

        self._define_parameters()
        self._initialize_parameters()

    def _define_parameters(self):
        """定义可学习参数：特征变换权重和小波滤波器。"""
        # 小波滤波器的对角索引（仅对角元素）
        diag_indices = torch.arange(
            self.num_nodes, dtype=torch.long, device=self.device
        )
        self.register_buffer(
            "diagonal_indices", torch.stack([diag_indices, diag_indices], dim=0)
        )

        initial_filter = torch.ones(
            self.num_nodes, 1, dtype=torch.double, device=self.device
        )
        self.diagonal_filter = nn.Parameter(initial_filter)

    def _initialize_parameters(self):
        """初始化可学习参数。"""
        nn.init.ones_(self.diagonal_filter)  # 初始化滤波器接近恒等

    def forward(
        self,
        wavelet_indices,
        wavelet_values,
        wavelet_inv_indices,
        wavelet_inv_values,
        features,
    ):
        """
        前向传播：执行基于小波的过滤和特征变换。

        :param wavelet_indices: 小波矩阵非零元素的索引 (Phi)。
        :param wavelet_values: 小波矩阵非零元素的值。
        :param wavelet_inv_indices: 反转小波矩阵非零元素的索引 (Phi^{-1})。
        :param wavelet_inv_values: 反转小波矩阵非零元素的值。
        :param features: 形状为 [batch_size, num_nodes, input_dim] 的输入节点特征。
        :return: 过滤和变换后的输出特征。
        """
        batch_size, num_nodes, _ = features.shape

        # ! please note that torch_sparse.spspmm() do not support autograd
        # see https://github.com/rusty1s/pytorch_sparse/issues/45#issuecomment-2809747932

        # 构建稀疏的小波矩阵和反转小波矩阵
        Phi_inv = dglsp.spmatrix(
            wavelet_inv_indices,
            wavelet_inv_values,
            shape=(self.num_basis, self.num_nodes),  # 通常是 (N, N)
        )
        Phi = dglsp.spmatrix(
            wavelet_indices, wavelet_values, shape=(self.num_nodes, self.num_basis)
        )

        # 创建对角滤波器矩阵: diag(θ) 形状为 [N, N]
        filter_diag = dglsp.diag(
            self.diagonal_filter.view(-1), shape=(self.num_nodes, self.num_nodes)
        )

        # 计算: Phi_inv @ diag(θ) @ Phi
        rescaled_phi = dglsp.spspmm(Phi_inv, filter_diag)  # [N, N]
        diffusion_op = dglsp.spspmm(rescaled_phi, Phi)  # [N, N]

        # 应用特征变换和扩散
        outputs = torch.zeros(
            batch_size, num_nodes, self.output_dim, device=self.device
        )
        for b in range(batch_size):
            signal_b = features[b, :, :]
            outputs[b, :, :] = dglsp.spmm(diffusion_op, signal_b)

        return outputs
