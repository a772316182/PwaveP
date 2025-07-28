import argparse
import os
import sys

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
    if basename == "wavetlet_def" or project_root_path == "/":
        break
    project_root_path = parent_path
print("auto detect project root path: ", project_root_path)
sys.path.append(project_root_path)
sys.path.append(os.path.join(project_root_path, "src"))

if __name__ == "__main__":
    from src.models.trainer import load_model, load_dataset
    from src.utils.auto_gpu_chose import AutoGPUChoseManager
    from src.datasets.attacked_data import AttackedData
    from src.defenders.fourier_low_filter.starter import start_gsp_low_filter
    from src.utils.mongo_log import MongoDBLog

    mongo_logger = MongoDBLog("exp_data")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=16, help="batch size in training"
    )
    args = parser.parse_args()
    defender_name = 'fourier'
    for attacker_name in ["gsda", "eidos", "geoa3", "hit_adv", "si_adv"]:
        for dataset_name in ["ModelNet40", "ShapeNetPart"]:
            for model_name in ["dgcnn", "pointnet", "pointnet2", "curvenet"]:
                args.attacker =  attacker_name
                args.model = model_name
                args.dataset = dataset_name
                args.defender = defender_name
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

                loader_train, loader_test = load_dataset(args.batch_size, args.dataset, data_root)
                logger.info("Eval on clean data")
                model.eval()
                attacked_data = torch.load(
                    f"{project_root_path}/attacked_data/{args.attacker}/{args.dataset}_{args.model}_res.pt",
                    map_location=device,
                )
                attack_loader = DataLoader(
                    AttackedData(attacked_data),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                def_res = start_gsp_low_filter(
                    victim_loader=attack_loader,
                    device=device,
                    model=model,
                )

                def_acc = round(def_res["acc"] * 100, 2)
                logger.info(f"{args.attacker} {args.model} {args.dataset}")
                logger.info(f"{args.defender} def result: {def_acc}")
                mongo_logger.insert_new("run_fourier_batched", {
                    "defender": args.defender,
                    "attacker": args.attacker,
                    "model": args.model,
                    "dataset": args.dataset,
                    "acc": def_acc
                })

