import dgl
import numpy as np
import torch
from dgl import DGLGraph
from scipy import sparse
from torch import Tensor
from torch import sparse_coo_tensor
from torch.nn.functional import normalize


def chunked_large_dense_to_sparse_coo(dense_matrix, chunk_size=1000):
    """
    Convert a very large dense matrix to a sparse COO tensor using chunked processing.

    :param dense_matrix: A very large dense matrix (e.g., numpy array).
    :param chunk_size: The size of each chunk for processing.
    :return: A sparse COO tensor in PyTorch.
    """
    num_rows, num_cols = dense_matrix.shape
    all_values = []
    all_rows = []
    all_cols = []

    for i in range(0, num_rows, chunk_size):
        for j in range(0, num_cols, chunk_size):
            chunk = dense_matrix[i : i + chunk_size, j : j + chunk_size]
            sparse_chunk = sparse.coo_matrix(chunk.cpu())

            if sparse_chunk.nnz > 0:
                values = torch.tensor(sparse_chunk.data, dtype=torch.float32)
                rows = torch.tensor(sparse_chunk.row + i, dtype=torch.long)
                cols = torch.tensor(sparse_chunk.col + j, dtype=torch.long)

                all_values.append(values)
                all_rows.append(rows)
                all_cols.append(cols)

    combined_values = torch.cat(all_values)
    combined_rows = torch.cat(all_rows)
    combined_cols = torch.cat(all_cols)
    indices = torch.stack([combined_rows, combined_cols])

    sparse_tensor = torch.sparse_coo_tensor(
        indices, combined_values, size=(num_rows, num_cols), device=dense_matrix.device
    )
    return sparse_tensor.coalesce()


class SpectralGraphWavelets(object):
    """
    Object to sparsify the wavelet coefficients for a graph.
    """

    def __init__(
        self,
        graph,
        graph_laplacian,
        scaling_tau=5,
        wavelets_tau=[0.02, 0.15],
        num_wavelets=3,
        approximation_order=3,
        tolerance=0,
        trace_func=print,
        device="cpu",
    ):
        """
        :param graph_laplace: Laplace of graph object.
        :param scale: Kernel scale length parameter.
        :param approximation_order: Chebyshev polynomial order.
        :param tolerance: Tolerance for sparsification.
        """
        self.device = device
        self.graph: DGLGraph = graph
        self.lmax_graph_laplacian: float = self.graph_laplacian_eigenvalue_max()

        self.graph_laplacian: Tensor = graph_laplacian
        self.num_nodes_graph: int = self.graph_laplacian.shape[0]

        self.num_scales: int = num_wavelets + 1
        self.scaling_tau: int = scaling_tau
        self.wavelets_tau: Tensor = torch.linspace(
            wavelets_tau[0], wavelets_tau[1], num_wavelets
        )

        self.approximation_order: int = approximation_order
        self.tolerance: float = tolerance
        self.trace_func = trace_func

        self.wavelet_matrices = []

    def calculate_wavelet_numpy(self):
        # 这是普通特征分解的版本，不过切比雪夫够用了，仅供检查
        eigenvalues, eigenvectors = np.linalg.eigh(
            self.graph_laplacian.to_dense().cpu().numpy()
        )
        aa = [
            np.diag(self.filter[i](torch.tensor(eigenvalues)))
            for i in range(len(self.filter))
        ]
        bb = [eigenvectors @ aaa @ eigenvectors.T for aaa in aa]
        cc = np.row_stack(bb)
        return cc

    def calculate_wavelet(self, compute_error=False):
        """
        Creating sparse wavelets.
        :return remaining_waves: Sparse matrix of attenuated wavelets.
        """
        impulse = torch.eye(
            self.num_nodes_graph, dtype=torch.float32, device=self.device
        )
        wavelet_coefficients = self.cheby_op(
            self.graph_laplacian, self.chebyshev, impulse
        )

        wavelet_coefficients[wavelet_coefficients < self.tolerance] = 0
        ind_1, ind_2 = wavelet_coefficients.nonzero(as_tuple=True)

        # shape of remaining_waves: (num_nodes_graph * self.num_scales, num_nodes_graph)
        values = wavelet_coefficients[ind_1, ind_2]
        indices = torch.stack([ind_1, ind_2])
        size = (self.num_nodes_graph * self.num_scales, self.num_nodes_graph)
        remaining_waves = sparse_coo_tensor(indices, values, size, dtype=torch.float32)
        if compute_error:
            real_eig_waves = self.calculate_wavelet_numpy()
            error = torch.tensor(real_eig_waves, device=self.device) - remaining_waves
            self.trace_func("Error (SUM): {}".format(error.sum()))
            self.trace_func("Error (AVG): {}".format(error.mean()))
            self.trace_func("Error (MAX): {}".format(error.max()))
            self.trace_func("Error (MIN): {}".format(error.min()))

        return remaining_waves.coalesce()

    def normalize_matrices(self):
        """
        Normalizing the wavelet and inverse wavelet matrices.
        """
        self.trace_func("Normalizing the sparsified wavelets.")
        for i, phi_matrix in enumerate(self.wavelet_matrices):
            self.wavelet_matrices[i] = normalize(
                phi_matrix.to_dense(), p=1, dim=1
            ).to_sparse()

    def calculate_density(self):
        """
        Calculating the density of the sparsified wavelet matrices.
        """
        wavelet_density = len(self.wavelet_matrices[0].coalesce().indices()[0]) / (
            self.num_nodes_graph * self.num_scales * self.num_nodes_graph
        )
        wavelet_density = str(round(100 * wavelet_density, 2))
        inverse_wavelet_density = len(
            self.wavelet_matrices[1].coalesce().indices()[0]
        ) / (self.num_nodes_graph * self.num_scales * self.num_nodes_graph)
        inverse_wavelet_density = str(round(100 * inverse_wavelet_density, 2))
        self.trace_func("Density of wavelets: " + wavelet_density + "%.")
        self.trace_func(
            "Density of inverse wavelets: " + inverse_wavelet_density + "%."
        )

    def calculate_all_wavelets(self, compute_error=False):
        """
        Graph wavelet coefficient calculation.
        """
        self.wavelet_matrices = []
        self.trace_func("Wavelet calculation and sparsification started.")
        self.trace_func(
            "Max eigenvalue of graph laplacian:{}".format(
                self.lmax_graph_laplacian.item()
            )
        )

        self.filter = self.filter(
            graph_eigenvalue_max=self.lmax_graph_laplacian, tau=self.wavelets_tau
        )

        # ------------- Computing wavelets --------------#
        self.chebyshev = self.compute_cheby_coeff(
            self.filter,
            num_filter=self.num_scales,
            graph_eigenvalue_max=self.lmax_graph_laplacian,
            m=self.approximation_order,
        )

        sparsified_wavelets = self.calculate_wavelet(compute_error=compute_error)

        self.wavelet_matrices.append(sparsified_wavelets)

        # --------- Computing inverse wavelets ------------#
        self.trace_func("Computing inverse wavelets")
        wavelets_inv = torch.linalg.pinv(sparsified_wavelets.to_dense())
        self.trace_func("Finish computing inverse wavelets")
        remaining_wavelets_inv = torch.where(
            wavelets_inv < self.tolerance, 0, wavelets_inv
        )
        sparsified_wavelets_inv = chunked_large_dense_to_sparse_coo(
            remaining_wavelets_inv
        )
        self.wavelet_matrices.append(sparsified_wavelets_inv.coalesce())

        self.normalize_matrices()
        self.calculate_density()

    def graph_laplacian_eigenvalue_max(self):
        return dgl.laplacian_lambda_max(self.graph)[0]

    def filter(self, graph_eigenvalue_max, tau):
        g = []

        # Low pass filtering, scaling filter
        g.append(lambda x: torch.exp(-1 * self.scaling_tau * x / graph_eigenvalue_max))

        # Bass pass filtering, wavelet filter
        for t in tau:
            g.append(
                lambda x, t=t: torch.exp(
                    -1 * torch.pow((x / graph_eigenvalue_max - 0.5), 2) / (2 * t * t)
                )
            )
        return g

    def filter_heat(self, graph_eigenvalue_max, tau):
        g = []
        for t in tau:
            g.append(lambda x, t=t: torch.exp(-t * x / graph_eigenvalue_max))
        return g

    def compute_cheby_coeff(
        self,
        filter,
        num_filter=1,
        graph_eigenvalue_max=2,
        m=30,
        N=None,
        *args,
        **kwargs
    ):
        if not N:
            N = m + 1

        a_arange = [0, graph_eigenvalue_max]

        a1 = (a_arange[1] - a_arange[0]) / 2
        a2 = (a_arange[1] + a_arange[0]) / 2
        c = torch.zeros((num_filter, m + 1))

        tmpN = torch.arange(N)
        num = torch.cos(torch.pi * (tmpN + 0.5) / N)
        for i in range(num_filter):
            for o in range(m + 1):
                c[i][o] = (
                    2.0
                    / N
                    * torch.dot(
                        filter[i](a1 * num + a2),
                        torch.cos(torch.pi * o * (tmpN + 0.5) / N),
                    )
                )
        return c

    def cheby_op(self, graph_laplacian, c, signal, **kwargs):
        r"""
        Chebyshev polynomial of graph Laplacian applied to vector.

        Parameters
        ----------
        G : Graph Laplacian
        c : ndarray or list of ndarrays
            Chebyshev coefficients for a Filter or a Filterbank
        signal : ndarray
            Signal to filter

        Returns
        -------
        r : ndarray
            Result of the filtering

        """
        # Handle if we do not have a list of filters but only a simple filter in cheby_coeff.
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, device=self.device)

        c = torch.atleast_2d(c)
        Nscales, M = c.shape

        num_nodes = graph_laplacian.shape[0]

        try:
            Nv = signal.shape[1]
            r = torch.zeros((num_nodes * Nscales, Nv), device=self.device)
        except IndexError:
            r = torch.zeros((num_nodes * Nscales), device=self.device)

        lmax = self.graph_laplacian_eigenvalue_max()
        a_arange = [0, lmax]

        a1 = float(a_arange[1] - a_arange[0]) / 2.0
        a2 = float(a_arange[1] + a_arange[0]) / 2.0

        twf_old = signal  # T0=1
        twf_cur = (graph_laplacian.mm(signal) - a2 * signal) / a1  # T1=y

        tmpN = torch.arange(num_nodes, dtype=torch.int64, device=self.device)
        for i in range(Nscales):
            r[tmpN + num_nodes * i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur

        factor = (
            2 / a1 * (-a2 * torch.eye(num_nodes, device=self.device) + graph_laplacian)
        )
        for k in range(2, M):
            twf_new = factor.mm(twf_cur) - twf_old  # T2 = 2y*T1 - T0
            for i in range(Nscales):
                r[tmpN + num_nodes * i] += c[i, k] * twf_new

            twf_old = twf_cur
            twf_cur = twf_new

        return r
