import numpy
import torch
from torch_geometric import utils
from torch_geometric.nn import knn_graph


def generate_onehot_signal_for_node_i(i, num_nodes, use_pytorch=False):
    if use_pytorch:
        signal = torch.zeros(num_nodes)
    else:
        signal = numpy.zeros(num_nodes)
    signal[i] = 1
    return signal


def build_adjacency_matrix_from_batched_point_clouds(
    point_clouds, k=10, return_numpy=False, spare_tensor=False
):
    batch_size = point_clouds.shape[0]
    device = point_clouds.device

    adj_matrices = []
    for i in range(batch_size):
        coordinates = point_clouds[i]
        edge_index = knn_graph(coordinates, k=k)
        edge_index = utils.to_undirected(edge_index)
        A = utils.to_torch_coo_tensor(edge_index).to(device)
        if return_numpy:
            A = A.cpu().to_dense().numpy()
        elif not spare_tensor:
            A = A.to_dense()
        adj_matrices.append(A)
    return adj_matrices


def sparse_normalized_laplacian(adj_matrix):
    batch_size, n = adj_matrix.size(0), adj_matrix.size(1)
    laplacians = []
    for i in range(batch_size):
        adj_item = adj_matrix[i]
        edge_index, _ = utils.to_edge_index(adj_item)
        L = utils.get_laplacian(edge_index, normalization="sym")
        L = torch.sparse_coo_tensor(L[0], L[1])
        laplacians.append(L)
    stacked_L = torch.stack(laplacians, dim=0)
    return stacked_L


def get_fourier_base(pc, k=20):
    adj_matrices = build_adjacency_matrix_from_batched_point_clouds(
        pc, k=k, spare_tensor=True
    )
    eigvals, eigvecs = torch.linalg.eigh(
        sparse_normalized_laplacian(torch.stack(adj_matrices, dim=0)).to_dense()
    )  # shape [B, N, N]
    return eigvals, eigvecs