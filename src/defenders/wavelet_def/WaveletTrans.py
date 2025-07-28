import numpy
import pygsp2
import pygsp2.graphs
import torch.nn as nn
from loguru import logger
from torch import Tensor

from src.defenders.wavelet_def.utils import *
from src.utils.graph import build_adjacency_matrix_from_batched_point_clouds, sparse_normalized_laplacian


class WaveletTransformUtil(nn.Module):
    def __init__(self, batched_pc: Tensor, filter_name: str, num_wavelets: int, k_neighbors: int = 20):
        """
        Args:
            batched_pc (Tensor): 点云
            filter_name (str): 小波滤波器名称 ('meyer', 'mexicanhat', 'heat').
            num_wavelets (int): 要使用的小波频带（尺度）数量。
            k_neighbors (int): 构建k-NN图时使用的邻居数。
        """
        super().__init__()
        self.batched_pc = batched_pc
        self.filter_name = filter_name
        self.num_wavelets = num_wavelets
        self.k_neighbors = k_neighbors

        # 用于缓存计算好的小波基，避免重复计算
        self.bases_built = False
        self.Us, self.wavelets_fs, self.inv_wavelets_fs = None, None, None
        self.coffs = None
        self._build_bases(batched_pc)

    def transform(self, signal_batch: torch.Tensor) -> torch.Tensor:
        """对信号进行图小波正向变换。"""
        spec_batch = torch.einsum('bne,bnc->bec', self.Us, signal_batch)
        spec_w_batch = spec_batch.unsqueeze(-1) * self.wavelets_fs.unsqueeze(2)
        coeffs_batch = torch.einsum('bne,becf->bncf', self.Us, spec_w_batch)
        self.coffs = coeffs_batch
        return coeffs_batch

    def inverse_transform(self, coeffs_batch: torch.Tensor) -> torch.Tensor:
        """对小波系数进行逆向变换以重构信号。"""
        # 对于实对称图，U_inv 就是 U
        spec2_batch = torch.einsum('bne,bncf->becf', self.Us, coeffs_batch)
        weighted_spec2_batch = (spec2_batch * self.inv_wavelets_fs.unsqueeze(2)).sum(dim=3)
        inv_signal_batch = torch.einsum('bne,bec->bnc', self.Us, weighted_spec2_batch)
        return inv_signal_batch

    def extract_grad_wrt_coffs(self, target_model: nn.Module, signal_batch: torch.Tensor):
        coeffs_batch = self.transform(signal_batch).requires_grad_()
        inv_signal_batch = self.inverse_transform(coeffs_batch)
        # --- Error Check ---
        # Calculate error per sample in the batch, then average, or overall norm
        error_per_sample = torch.linalg.norm(signal_batch - inv_signal_batch,
                                             dim=(1, 2))  # Frobenius norm for each [N,C] matrix
        mean_error = error_per_sample.mean().item()
        total_norm_error = torch.norm(signal_batch - inv_signal_batch).item()
        if total_norm_error > 1e-10:
            logger.warning("[wavelet error] wavelet error is too large")
            logger.warning(f"[mean_error] wavelet error is {mean_error:.3e}")
            logger.warning(f"[total_norm_error] wavelet error is {total_norm_error:.3e}")

        logits = target_model(inv_signal_batch.float())
        loss = torch.nn.functional.cross_entropy(
            logits["logit"],
            logits["logit"].argmax(dim=-1).clone().detach()
        )
        grad_of_coffs = torch.autograd.grad(loss, coeffs_batch, retain_graph=True)[0]
        return grad_of_coffs, coeffs_batch

    def _wavelet_factory(self, g: pygsp2.graphs.Graph):
        """根据名称创建 pygsp2 小波滤波器实例。"""
        if self.filter_name.lower() == "meyer":
            return pygsp2.filters.Meyer(G=g, Nf=self.num_wavelets)
        elif self.filter_name.lower() == "heat":
            # 对于Heat小波，需要先估计lmax来确定尺度
            g.estimate_lmax()
            scales = g.lmax * np.logspace(-2, 0, self.num_wavelets)
            return pygsp2.filters.Heat(G=g, scale=scales)
        elif self.filter_name.lower() == "mexicanhat":
            return pygsp2.filters.MexicanHat(G=g, Nf=self.num_wavelets)
        else:
            raise ValueError(f"Unknown filter name: {self.filter_name}")

    def _build_bases(self, batched_pc: torch.Tensor):
        """
        根据输入的点云批量构建并缓存小波变换所需的基矩阵。
        这是一个计算密集型操作。
        """
        device = batched_pc.device
        logger.info(f"Building wavelet bases for {batched_pc.shape[0]} graphs on device {device}...")

        adj_matrices = build_adjacency_matrix_from_batched_point_clouds(
            batched_pc, k=self.k_neighbors, return_numpy=False, spare_tensor=True
        )
        adjs = [item.to_dense().cpu().numpy() for item in adj_matrices]

        laplacians = sparse_normalized_laplacian(torch.stack(adj_matrices, dim=0)).double()
        eigvals, eigvecs = torch.linalg.eigh(laplacians.to_dense())
        eigvals = eigvals.cpu().numpy()
        eigvecs = eigvecs.cpu().numpy()

        Us_list, wavelets_fs_list, inv_wavelets_fs_list = [], [], []

        for i, adj in enumerate(adjs):
            pygsp2_g = pygsp2.graphs.Graph(adj)
            pygsp2_g.compute_laplacian(lap_type="normalized")
            # pygsp2_g.compute_fourier_basis() # scipy too slow!
            pygsp2_g._U = eigvecs[i]
            pygsp2_g._e = eigvals[i]
            pygsp2_g.estimate_lmax()

            pygsp2_filter = self._wavelet_factory(pygsp2_g)
            pygsp2_filter_inv = pygsp2_filter.inverse()

            U = torch.from_numpy(pygsp2_filter.G.U).to(device)
            wavelets_f = torch.from_numpy(pygsp2_filter.evaluate(pygsp2_filter.G.e)).to(device).T
            inv_wavelets_f = torch.from_numpy(pygsp2_filter_inv.evaluate(pygsp2_filter_inv.G.e)).to(device).T

            Us_list.append(U)
            wavelets_fs_list.append(wavelets_f)
            inv_wavelets_fs_list.append(inv_wavelets_f)

        self.Us = torch.stack(Us_list).double()
        self.wavelets_fs = torch.stack(wavelets_fs_list).double()
        self.inv_wavelets_fs = torch.stack(inv_wavelets_fs_list).double()
        self.bases_built = True
