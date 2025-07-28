import argparse
import os
import sys

import torch

project_root_path = os.path.dirname(os.path.abspath(__file__))
while True:
    parent_path = os.path.dirname(project_root_path)
    if os.path.basename(project_root_path) == "wavetlet_def":
        break
    project_root_path = parent_path
print("auto detect project root path: ", project_root_path)
sys.path.append(project_root_path)

if __name__ == "__main__":
    from src.models.trainer import train_model
    from src.utils.auto_gpu_chose import AutoGPUChoseManager

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="pointnet2",
        help="model name",
        choices=["pointnet", "dgcnn", "pointnet2", "curvenet", "pct", "pointmlp"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ShapeNetPart",
        help="dataset name",
        choices=["ModelNet40", "ShapeNetPart"],
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size in training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=200, help="number of epochs to train"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="learning rate for training"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0002, help="weight decay"
    )
    parser.add_argument(
        "--scheduler_step", type=int, default=100, help="scheduler step"
    )
    parser.add_argument(
        "--scheduler_gamma", type=float, default=0.2, help="scheduler gamma"
    )
    args = parser.parse_args()
    device = (
        f"cuda:{AutoGPUChoseManager().auto_choice()}"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_root = os.path.join(project_root_path, "data")
    ckpt_root = os.path.join(project_root_path, "ckpt")

    model = train_model(
        device=device,
        model_name=args.model,
        dataset_name=args.dataset,
        data_root=data_root,
        ckpt_root=ckpt_root,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_step=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
    )
