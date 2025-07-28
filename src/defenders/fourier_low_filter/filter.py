from typing import Union

import torch
from torch import Tensor

from src.utils.graph import build_adjacency_matrix_from_batched_point_clouds, sparse_normalized_laplacian


def low_filter(pc: torch.Tensor, k=10, cutoff=0.67, signal=None) -> Union[tuple[Tensor, Tensor], Tensor]:
    batch_size, num_points, _ = pc.shape
    adj_matrices = build_adjacency_matrix_from_batched_point_clouds(pc, k=k,
                                                                    spare_tensor=True, return_numpy=False)
    adj_matrices = torch.stack(adj_matrices, dim=0)  # (batch_size, n, n)

    # Compute normalized laplacians for all point clouds in the batch
    Ls = sparse_normalized_laplacian(adj_matrices)

    # Eigen decomposition for all laplacians
    eigvals, eigvecs = torch.linalg.eigh(Ls.to_dense())

    max_eigvals = torch.max(eigvals, dim=1, keepdim=True)[0]
    min_eigvals = torch.min(eigvals, dim=1, keepdim=True)[0]
    low_pass_filters = torch.diag_embed((eigvals < cutoff).float())

    # Apply low pass filter to each dimension
    filtered_pcs = torch.zeros_like(pc)
    t = torch.bmm(torch.bmm(eigvecs, low_pass_filters), eigvecs.transpose(1, 2))
    for _ in range(3):  # 对每个维度(x, y, z)进行滤波
        filtered_pcs[:, :, _] = torch.bmm(
            t,
            pc[:, :, _].unsqueeze(2),
        ).squeeze(2)

    if signal:
        filtered_signal = torch.zeros_like(signal)
        if len(signal.shape) == 1:
            signal = signal.unsqueeze(0)

        for _ in range(signal.shape[0]):
            filtered_signal[:, :, _] = torch.bmm(
                t,
                signal[:, :, _].unsqueeze(2),
            ).squeeze(2)

        return filtered_pcs, filtered_signal
    else:
        return filtered_pcs
