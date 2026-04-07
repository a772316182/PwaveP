import argparse
import os
import sys

import dgl
import networkx
import pytorch3d.loss
import torch
import torch_geometric
from loguru import logger
from torch.utils.data import DataLoader

torch_geometric.seed_everything(0)
project_root_path = os.path.dirname(os.path.abspath(__file__))
while True:
    parent_path = os.path.dirname(project_root_path)
    basename = os.path.basename(project_root_path)
    print("project root path: ", project_root_path)
    if basename == "pwavep" or project_root_path == "/":
        break
    project_root_path = parent_path
print("auto detect project root path: ", project_root_path)
sys.path.append(project_root_path)
sys.path.append(os.path.join(project_root_path, "src"))

if __name__ == "__main__":
    from src.models.trainer import load_model
    from src.utils.auto_gpu_chose import AutoGPUChoseManager
    from src.models.evaluator import validate_adv_examples
    from src.datasets.attacked_data import AttackedData

    from src.trainable_wavelet_model.SpectralGraphWavelets import SpectralGraphWavelets
    from src.trainable_wavelet_model.layer import GraphWaveletLayer  # 修改导入的类名
    from src.utils.graph import (
        build_adjacency_matrix_from_batched_point_clouds,
        sparse_normalized_laplacian,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attacker",
        type=str,
        default="si_adv",
        help="attacker name",
        choices=["gsda", "eidos", "geoa3", "hit_adv", "si_adv"],
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dgcnn",
        help="model name",
        choices=["pointnet", "dgcnn", "pct", "curvenet", "pointnet2"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ModelNet40",
        help="dataset name",
        choices=["ModelNet40", "ShapeNetPart"],
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size in training"
    )
    args = parser.parse_args()

    args.dataset = args.dataset.lower()
    args.model = args.model.lower()

    device = (
        f"cuda:{AutoGPUChoseManager().auto_choice()}"
        if torch.cuda.is_available()
        else "cpu"
    )
    data_root = os.path.join(project_root_path, "data")
    ckpt_root = os.path.join(project_root_path, "ckpt")

    model = load_model(
        device,
        args.model,
        args.dataset,
        ckpt_root,
    )

    attacked_data = torch.load(
        f"{project_root_path}/attacked_data/{args.attacker}/{args.dataset}_{args.model}_res.pt",
        map_location=device,
    )

    clean_acc, asr = validate_adv_examples(attacked_data, model, device)

    logger.info(
        f"{args.attacker} at model {args.model} and dataset {args.dataset} attack result:"
    )
    logger.info(f"clean acc: {clean_acc['acc']}")
    logger.info(f"adv acc: {asr['acc']}")

    attack_loader = DataLoader(
        AttackedData(attacked_data),
        batch_size=args.batch_size,
        shuffle=False,
    )

    for i, data_item in enumerate(attack_loader):
        clean_pc = data_item["real_data"]
        adv_pc = data_item["attacked_data"]

        adv_adj_matrices = build_adjacency_matrix_from_batched_point_clouds(
            adv_pc, k=20, return_numpy=False, spare_tensor=True
        )
        adv_adjs = [item.to_dense().cpu().numpy() for item in adv_adj_matrices]

        adv_laplacians = sparse_normalized_laplacian(
            torch.stack(adv_adj_matrices, dim=0)
        ).float()

        """
        for demo, just using #0 pc
        """

        adv_pc_item = torch.unsqueeze(adv_pc[0], 0).float()
        clean_pc_item = torch.unsqueeze(clean_pc[0], 0).float()

        demo_g = dgl.from_networkx(networkx.Graph(adv_adjs[0]))
        pc_graph_sparse_wavelets = SpectralGraphWavelets(
            graph=demo_g.clone().to(device),
            graph_laplacian=adv_laplacians[0],
            scaling_tau=1,
            wavelets_tau=[0.01, 1],
            num_wavelets=30,
            approximation_order=50,
            device=device,
            tolerance=1e-4,
            trace_func=logger.debug,
        )
        pc_graph_sparse_wavelets.calculate_all_wavelets(compute_error=False)
        pc_graph_wavelet_matrices = pc_graph_sparse_wavelets.wavelet_matrices[0].float()
        pc_graph_inverse_wavelet_matrices = pc_graph_sparse_wavelets.wavelet_matrices[
            1
        ].float()

        # 修改模型初始化部分
        model_wavelet = GraphWaveletLayer(
            input_dim=adv_pc_item.shape[-1],
            output_dim=adv_pc_item.shape[-1],
            wavelet_shape=pc_graph_wavelet_matrices.shape,
            dropout_rate=0.5,
            device=device,
        ).to(device)

        optimizer = torch.optim.NAdam(model_wavelet.parameters(), lr=1e-2)
        wavelet_matrices_str = [
            pc_graph_wavelet_matrices.to_dense().detach().cpu().numpy(),
            pc_graph_inverse_wavelet_matrices.to_dense().detach().cpu().numpy(),
        ]
        print(f"{'Epoch':<10}{'Loss':<20}{'Filter Mean':<20}")
        print("-" * 50)  # 打印分割线
        for j in range(10):
            model_wavelet.train()
            optimizer.zero_grad()

            filtered_pc = model_wavelet.forward(
                wavelet_indices=torch.LongTensor(wavelet_matrices_str[0].nonzero()).to(
                    device
                ),
                wavelet_values=torch.tensor(
                    wavelet_matrices_str[0][wavelet_matrices_str[0].nonzero()],
                    dtype=torch.float64,
                )
                .view(-1)
                .to(device),
                wavelet_inv_indices=torch.LongTensor(
                    wavelet_matrices_str[1].nonzero()
                ).to(device),
                wavelet_inv_values=torch.tensor(
                    wavelet_matrices_str[1][wavelet_matrices_str[1].nonzero()],
                    dtype=torch.float64,
                )
                .view(-1)
                .to(device),
                features=adv_pc_item.double(),
            )
            # for test: just recover the attacked pc, so using chamfer distance as loss
            loss = pytorch3d.loss.chamfer_distance(filtered_pc, clean_pc_item)[0]

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            filter_mean_val = torch.mean(model_wavelet.diagonal_filter).item()
            print(f"{j + 1:<10}{loss_val:<20.6f}{filter_mean_val:<20.6f}")
