import numpy
import pygsp2
import pygsp2.graphs
import torch.nn as nn
from loguru import logger
from torch import Tensor

from src.defenders.wavelet_def.utils import *
from src.utils.graph import (
    build_adjacency_matrix_from_batched_point_clouds,
    sparse_normalized_laplacian,
    torch_sparse_to_scipy,
)


class WaveletTransformUtilCheb(nn.Module):
    def __init__(
        self,
        device: str,
        batched_pc: Tensor,
        filter_name: str,
        num_wavelets: int,
        k_neighbors: int = 20,
        approximation_order: int = 30,
    ):
        super().__init__()
        self.device = device
        self.approximation_order = approximation_order
        self.batched_pc = batched_pc
        self.batch_size = batched_pc.shape[0]
        self.num_nodes = batched_pc.shape[1]
        self.filter_name = filter_name
        self.num_wavelets = num_wavelets
        self.k_neighbors = k_neighbors

        # --- 初始化部分 (无变化) ---
        self.adjs = torch.stack(
            build_adjacency_matrix_from_batched_point_clouds(
                batched_pc, k=self.k_neighbors, return_numpy=False, spare_tensor=True
            ),
            dim=0,
        )
        self.laplacians = sparse_normalized_laplacian(self.adjs).to(device).double()

        pygsp2_gs = [
            pygsp2.graphs.Graph(
                adjacency=torch_sparse_to_scipy(self.adjs[i]), lap_type="normalized"
            )
            for i in range(self.batch_size)
        ]
        [item.estimate_lmax() for item in pygsp2_gs]
        self.lmax = torch.tensor(
            [g.lmax for g in pygsp2_gs], device=self.device, dtype=torch.double
        )

        self.filters = [self._wavelet_factory(item) for item in pygsp2_gs]
        cheb_coeffs_list = [
            self._compute_cheby_coeff_vectorized(item, m=self.approximation_order)
            for item in self.filters
        ]
        self.cheb_coffs_of_filters = torch.from_numpy(np.array(cheb_coeffs_list)).to(
            device=device, dtype=torch.double
        )

        self.inverse_filters = [item.inverse() for item in self.filters]
        inverse_cheb_coeffs_list = [
            self._compute_cheby_coeff_vectorized(item, m=self.approximation_order)
            for item in self.inverse_filters
        ]
        self.cheb_coffs_of_inverse_filters = torch.from_numpy(
            np.array(inverse_cheb_coeffs_list)
        ).to(device=device, dtype=torch.double)

        # --- 关键修正：手动构建块对角拉普拉斯算子 ---
        # torch.block_diag 对稀疏 CUDA 张量支持不佳，会引发 NotImplementedError。
        # 我们通过手动拼接索引和值来创建等效的块对角矩阵，这是一个更可靠的解决方法。
        all_indices = []
        all_values = []
        current_offset = 0
        for L in self.laplacians:
            # 确保是 COO 格式并获取索引和值
            L_coalesced = L.coalesce()
            indices = L_coalesced.indices()
            values = L_coalesced.values()

            # 为索引添加偏移量
            indices_offset = indices + current_offset
            all_indices.append(indices_offset)
            all_values.append(values)

            # 更新下一个矩阵的偏移量
            current_offset += self.num_nodes

        final_indices = torch.cat(all_indices, dim=1)
        final_values = torch.cat(all_values, dim=0)
        size = self.batch_size * self.num_nodes
        # 创建最终的、大的块对角稀疏矩阵
        self.L_block_diag = torch.sparse_coo_tensor(
            final_indices, final_values, (size, size)
        ).to(self.device)

    def transform(self, signal_batch: torch.Tensor) -> torch.Tensor:
        """完全向量化的小波变换 (分析过程)。无任何 for 循环。"""
        coeffs_batch = self._cheby_analysis_op_batched(signal_batch.double())
        return coeffs_batch.float()

    def inverse_transform(self, coeffs_batch: torch.Tensor) -> torch.Tensor:
        """向量化的小波逆变换 (合成过程)。循环仅在小波尺度上，内部操作完全批处理化。"""
        inv_signal_batch = self._cheby_synthesis_op_batched(coeffs_batch.double())
        return inv_signal_batch.float()

    def _cheby_op_batched(self, signal: Tensor, cheb_coeffs: Tensor) -> Tensor:
        """
        核心的完全批处理化切比雪夫递归。
        - signal: 信号张量, [B, N, F]
        - cheb_coeffs: 切比雪夫系数, [B, M]
        - returns: 应用滤波器后的信号, [B, N, F]
        """
        B, N, F = signal.shape
        M = cheb_coeffs.shape[1]

        # Reshape lmax for broadcasting: [B] -> [B, 1, 1]
        lmax_b = self.lmax.view(B, 1, 1)
        a1 = lmax_b / 2.0
        a2 = lmax_b / 2.0

        # --- 切比雪夫递归初始化 ---
        twf_old = signal  # T_0, Shape: [B, N, F]

        # Reshape for block diagonal multiplication: [B, N, F] -> [B*N, F]
        signal_reshaped = signal.reshape(B * N, F)
        # Apply block diagonal L: (B*N, B*N) @ (B*N, F) -> (B*N, F)
        L_x = torch.sparse.mm(self.L_block_diag, signal_reshaped)
        # Reshape back: [B*N, F] -> [B, N, F]
        L_x = L_x.view(B, N, F)

        twf_cur = (L_x - a2 * signal) / a1  # T_1, Shape: [B, N, F]

        # --- 结果累加 ---
        # Reshape coeffs for broadcasting: [B, M] -> [B, 1, 1, M]
        c = cheb_coeffs.view(B, 1, 1, M)

        # k=0 和 k=1 项
        r = 0.5 * c[..., 0] * twf_old + c[..., 1] * twf_cur

        # --- 递归计算 k >= 2 的项 ---
        for k in range(2, M):
            # Reshape for block diagonal multiplication
            twf_cur_reshaped = twf_cur.reshape(B * N, F)
            # Apply block diagonal L
            L_twf_cur = torch.sparse.mm(self.L_block_diag, twf_cur_reshaped).view(
                B, N, F
            )

            twf_new = (2 / a1) * L_twf_cur - (2 * a2 / a1) * twf_cur - twf_old

            # 累加贡献
            r += c[..., k] * twf_new

            twf_old, twf_cur = twf_cur, twf_new

        return r

    def _cheby_analysis_op_batched(self, signal_batch: Tensor) -> Tensor:
        """
        批处理化的分析操作 (修正版)。
        - signal_batch: [B, N, F]
        - returns: [B, N, F, J]
        """
        B, N, F = signal_batch.shape
        J, M = self.num_wavelets, self.approximation_order

        # --- 切比雪夫递归 ---
        # 为广播重塑 lmax: [B] -> [B, 1, 1]
        lmax_b = self.lmax.view(B, 1, 1)
        a1 = lmax_b / 2.0
        a2 = lmax_b / 2.0

        twf_old = signal_batch  # T_0, Shape: [B, N, F]

        # 使用块对角矩阵进行批处理图卷积
        L_x = torch.sparse.mm(self.L_block_diag, twf_old.reshape(B * N, F)).view(
            B, N, F
        )
        twf_cur = (L_x - a2 * twf_old) / a1  # T_1, Shape: [B, N, F]

        # --- 结果累加 ---
        # 获取 k=0 和 k=1 的系数
        c0 = self.cheb_coffs_of_filters[:, :, 0]  # Shape: [B, J]
        c1 = self.cheb_coffs_of_filters[:, :, 1]  # Shape: [B, J]

        # 正确地重塑系数切片以进行广播
        # c0.view(B, 1, 1, J): [B, J] -> [B, 1, 1, J]
        # twf_old.unsqueeze(3): [B, N, F] -> [B, N, F, 1]
        # 广播乘法结果: [B, N, F, J]
        r = 0.5 * c0.view(B, 1, 1, J) * twf_old.unsqueeze(3) + c1.view(
            B, 1, 1, J
        ) * twf_cur.unsqueeze(3)

        # 循环计算 k >= 2 的项
        for k in range(2, M):
            L_twf_cur = torch.sparse.mm(
                self.L_block_diag, twf_cur.reshape(B * N, F)
            ).view(B, N, F)
            twf_new = (2 / a1) * L_twf_cur - (2 * a2 / a1) * twf_cur - twf_old

            # 获取当前 k 的系数
            ck = self.cheb_coffs_of_filters[:, :, k]  # Shape: [B, J]

            # 累加贡献，使用正确的广播
            r += ck.view(B, 1, 1, J) * twf_new.unsqueeze(3)

            twf_old, twf_cur = twf_cur, twf_new

        return r

    def _cheby_synthesis_op_batched(self, coeffs_batch: Tensor) -> Tensor:
        """
        批处理化的合成操作。
        - coeffs_batch: [B, N, F, J]
        - returns: [B, N, F]
        """
        B, N, F, J = coeffs_batch.shape
        reconstructed_signal = torch.zeros(
            B, N, F, device=self.device, dtype=torch.double
        )

        # 对小波尺度 s 循环，但内部操作完全批处理化
        for s in range(J):
            # 应用第 s 个逆滤波器到第 s 个系数上
            reconstructed_signal += self._cheby_op_batched(
                signal=coeffs_batch[..., s],  # [B, N, F]
                cheb_coeffs=self.cheb_coffs_of_inverse_filters[:, s, :],  # [B, M]
            )

        return reconstructed_signal

    def extract_grad_wrt_coffs(
        self, target_model: nn.Module, signal_batch: torch.Tensor
    ):
        # 此方法无需更改，因为它调用的是 transform 和 inverse_transform
        coeffs_batch = self.transform(signal_batch).requires_grad_()
        inv_signal_batch = self.inverse_transform(coeffs_batch)
        # --- Error Check ---
        error = torch.norm(signal_batch - inv_signal_batch)
        if error > 1e-10:
            logger.warning(
                f"[wavelet error] Reconstruction error is large: {error.item():.3e}"
            )

        logits = target_model(inv_signal_batch.float())
        loss = torch.nn.functional.cross_entropy(
            logits["logit"], logits["logit"].argmax(dim=-1).clone().detach()
        )
        grad_of_coffs = torch.autograd.grad(loss, coeffs_batch, retain_graph=True)[0]
        return grad_of_coffs, coeffs_batch

    def _wavelet_factory(self, g: pygsp2.graphs.Graph):
        """根据名称创建 pygsp2 小波滤波器实例。(无需改动)"""
        if self.filter_name.lower() == "meyer":
            return pygsp2.filters.Meyer(G=g, Nf=self.num_wavelets)
        elif self.filter_name.lower() == "heat":
            scales = g.lmax * numpy.logspace(-2, 0, self.num_wavelets)
            return pygsp2.filters.Heat(G=g, scale=scales)
        elif self.filter_name.lower() == "mexicanhat":
            return pygsp2.filters.MexicanHat(G=g, Nf=self.num_wavelets)
        else:
            raise ValueError(f"Unknown filter name: {self.filter_name}")

    def _compute_cheby_coeff_vectorized(self, f, m=30, N=None):
        r"""
        一次性计算 Filterbank 中所有滤波器的切比雪夫系数 (向量化版本)。

        Parameters
        ----------
        f : pygsp2.filters.Filter
            包含一个或多个滤波器的 Filterbank 对象。
        m : int
            要计算的切比雪夫系数的最大阶数 (默认 = 30)。
        N : int
            用于计算积分的网格点数 (默认 = 2 * m)。

        Returns
        -------
        c : ndarray
            形状为 (f.Nf, m + 1) 的切比雪夫系数矩阵。
        """
        G = f.G

        # 使用一个更稳定的默认值来保证精度
        if N is None:
            N = 2 * m

        # 1. 创建用于数值积分的切比雪夫网格点
        # arange 是 [0, lmax]
        a1 = G.lmax / 2.0
        a2 = G.lmax / 2.0
        # tmpN 的形状是 (N,)
        tmpN = np.arange(N)
        # x 是评估点，形状为 (N,)
        x = a1 * np.cos(np.pi * (tmpN + 0.5) / N) + a2

        # 2. 一次性评估所有 Nf 个滤波器在网格点 x 上的值
        # f.evaluate(x) 返回一个形状为 (f.Nf, N) 的矩阵
        evaluated_kernels = f.evaluate(x)

        # 3. 创建切比雪夫基矩阵 T，形状为 (m + 1, N)
        # o 的形状是 (m + 1, 1)
        o = np.arange(m + 1).reshape(-1, 1)
        # cheby_basis[o, j] = cos(pi * o * (j + 0.5) / N)
        # 利用 numpy 的广播机制一次性计算出整个矩阵
        cheby_basis = np.cos(np.pi * o * (tmpN + 0.5) / N)

        # 4. 通过一次矩阵乘法计算所有系数
        # (f.Nf, N) @ (N, m + 1) -> (f.Nf, m + 1)
        coeffs = (2.0 / N) * np.dot(evaluated_kernels, cheby_basis.T)

        return coeffs
