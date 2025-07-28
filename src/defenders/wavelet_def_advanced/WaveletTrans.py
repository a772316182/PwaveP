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
        """对信号进行图小波正向变换。(已适配数据增强)"""
        signal_b = signal_batch.shape[0]
        base_b = self.Us.shape[0]

        # --- 新增：适配数据增强的逻辑 ---
        if signal_b > base_b and signal_b % base_b == 0:
            step = signal_b // base_b
            # 将基矩阵复制step倍以匹配输入的批次大小
            Us = self.Us.unsqueeze(1).repeat(1, step, 1, 1).view(-1, *self.Us.shape[1:])
            wavelets_fs = self.wavelets_fs.unsqueeze(1).repeat(1, step, 1, 1).view(-1, *self.wavelets_fs.shape[1:])
        else:
            # 如果批次大小匹配，则直接使用
            Us = self.Us
            wavelets_fs = self.wavelets_fs
        # ------------------------------------

        spec_batch = torch.einsum('bne,bnc->bec', Us, signal_batch)
        spec_w_batch = spec_batch.unsqueeze(-1) * wavelets_fs.unsqueeze(2)
        coeffs_batch = torch.einsum('bne,becf->bncf', Us, spec_w_batch)

        # 注意：不再缓存 self.coffs，因为它可能对应增强前或增强后的数据
        # self.coffs = coeffs_batch
        return coeffs_batch

    def inverse_transform(self, coeffs_batch: torch.Tensor) -> torch.Tensor:
        """对小波系数进行逆向变换以重构信号。(已适配数据增强)"""
        coeffs_b = coeffs_batch.shape[0]
        base_b = self.Us.shape[0]

        # --- 新增：适配数据增强的逻辑 ---
        if coeffs_b > base_b and coeffs_b % base_b == 0:
            step = coeffs_b // base_b
            # 将基矩阵复制step倍以匹配输入的批次大小
            Us = self.Us.unsqueeze(1).repeat(1, step, 1, 1).view(-1, *self.Us.shape[1:])
            inv_wavelets_fs = self.inv_wavelets_fs.unsqueeze(1).repeat(1, step, 1, 1).view(-1,
                                                                                           *self.inv_wavelets_fs.shape[
                                                                                            1:])
        else:
            Us = self.Us
            inv_wavelets_fs = self.inv_wavelets_fs
        # ------------------------------------

        spec2_batch = torch.einsum('bne,bncf->becf', Us, coeffs_batch)
        weighted_spec2_batch = (spec2_batch * inv_wavelets_fs.unsqueeze(2)).sum(dim=3)
        inv_signal_batch = torch.einsum('bne,bec->bnc', Us, weighted_spec2_batch)

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

    def get_coeffs_and_grads_advanced(self, target_model: nn.Module, signal_batch: torch.Tensor,
                                      label: torch.Tensor, eps=0.05, step=5):
        """
        计算复合损失相对于小波系数的梯度。
        复合损失 = 分类损失 + feature_loss_weight * 特征层损失
        """
        # ------------------- 使用前向Hook捕获特征图 -------------------
        if target_model.__class__.__name__ == "PointNet":
            track_modules = ["model.feat.fstn", "model.feat.conv3"]
        elif target_model.__class__.__name__ == "DGCNN":
            track_modules = ["model.conv5"]
        elif target_model.__class__.__name__ == "Pct":
            track_modules = ["model.identity"]
        elif target_model.__class__.__name__ == 'PointNet2ClsMsg':
            track_modules = ["sa3.mlp_convs"]
        elif target_model.__class__.__name__ == 'CurveNet':
            track_modules = ['cic11.conv1']

        device = signal_batch.device

        # 步骤 0: 在增强前，先计算原始数据的小波系数，这是后续滤波需要修改的对象
        original_coeffs = self.transform(signal_batch.double())

        # 步骤 1: 平滑技术 - 复制和加噪
        replicated_data, batch_index = replicate_tensor(signal_batch, step)
        replicated_label = replicate_label(label, step)
        if step > 1 and eps > 0:
            replicated_data = replicated_data + torch.rand_like(replicated_data) * eps * 2 - eps

        # 步骤 2: Hook 和反向传播
        captured_grads = []

        def backward_hook(module, grad_input, grad_output):
            if grad_input[0] is not None:
                captured_grads.append(grad_input[0].detach())

        handles = [h.register_full_backward_hook(backward_hook) for h in
                   [attrgetter(m)(target_model) for m in track_modules]]

        coeffs_batch_replicated = self.transform(replicated_data.double()).requires_grad_()
        inv_signal_batch_replicated = self.inverse_transform(coeffs_batch_replicated)

        logits = target_model(inv_signal_batch_replicated.float())["logit"]
        loss = torch.nn.functional.cross_entropy(logits, replicated_label)

        target_model.zero_grad()
        loss.backward()

        for handle in handles:
            handle.remove()

        # 步骤 3: 计算并聚合风险分数
        risk_replicated = torch.norm(coeffs_batch_replicated.grad, p=2, dim=(2, 3))
        risk_replicated /= (torch.norm(risk_replicated, p=2, dim=-1, keepdim=True) + 1e-9)

        with torch.no_grad():
            for grad in captured_grads[::-1]:
                if grad.shape[1] != risk_replicated.shape[1]: grad = grad.permute(0, 2, 1)
                r = torch.norm(grad, p=2, dim=-1)
                r /= (torch.norm(r, p=2, dim=-1, keepdim=True) + 1e-9)
                r[torch.isnan(r)] = 0
                risk_replicated += r

        final_risk = sum_same_batch(risk_replicated, batch_index) / step if step > 1 else risk_replicated

        # 步骤 4: 聚合梯度本身，用于后续的智能滤波
        grad_replicated = coeffs_batch_replicated.grad
        smoothed_grad = sum_same_batch(grad_replicated, batch_index) / step if step > 1 else grad_replicated

        return final_risk, original_coeffs, smoothed_grad


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