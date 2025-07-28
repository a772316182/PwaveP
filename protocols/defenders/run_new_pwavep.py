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
    from src.models.evaluator import validate, validate_adv_examples
    from src.datasets.attacked_data import AttackedData
    from src.defenders.wavelet_def_advanced.starter import start_wavelet_gard_wrt_coff_energy
    from src.utils.mongo_log import MongoDBLog

    mongo_logger = MongoDBLog("exp_data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attacker",
        type=str,
        default="gsda",
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
        "--batch_size", type=int, default=8, help="batch size in training"
    )
    parser.add_argument(
        "--k_neighbors", type=int, default=20, help="number of neighbors for wavelet transform"
    )
    parser.add_argument(
        "--num_wavelets", type=int, default=12, help="number of wavelets"
    )
    parser.add_argument(
        "--k_bands_to_filter", type=int, default=1, help="number of bands to filter"
    )
    parser.add_argument(
        "--attenuation_factor", type=float, default=0.0, help="attenuation factor"
    )
    parser.add_argument(
        "--drop_rate", type=float, default=0.01, help="dropout rate"
    )
    parser.add_argument(
        "--filter_rate", type=float, default=0.09, help="filter rate"
    )
    parser.add_argument(
        "--geo_risk_weight", type=float, default=1.0, help="geometric risk weight"
    )
    parser.add_argument(
        "--spectral_risk_weight", type=float, default=1.0, help="spectral risk weight"
    )
    parser.add_argument(
        "--grad_smooth_step", type=int, default=5, help="gradient smoothing step"
    )
    parser.add_argument(
        "--grad_smooth_eps", type=float, default=0.05, help="gradient smoothing epsilon"
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

    loader_train, loader_test = load_dataset(args.batch_size, args.dataset, data_root)

    logger.info("Eval on clean data")
    model.eval()
    res = validate(
        loader=loader_test,
        model=model,
        device=device,
    )
    logger.info(res)

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

    def_res = start_wavelet_gard_wrt_coff_energy(
        loader=attack_loader,
        device=device,
        model=model,
        k_neighbors=args.k_neighbors,
        num_wavelets=args.num_wavelets,
        k_bands_to_filter=args.k_bands_to_filter,
        attenuation_factor=args.attenuation_factor,
        drop_rate=args.drop_rate,
        filter_rate=args.filter_rate,
        geo_risk_weight=args.geo_risk_weight,
        spectral_risk_weight=args.spectral_risk_weight,
        grad_smooth_step=args.grad_smooth_step,
        grad_smooth_eps=args.grad_smooth_eps
    )

    def_acc = round(def_res["acc"] * 100, 2)

    mongo_logger.insert_new("Ablation", {
        "attacker": args.attacker,
        "model": args.model,
        "dataset": args.dataset,
        "acc": def_acc,
        "k_neighbors": args.k_neighbors,
        "num_wavelets": args.num_wavelets,
        "k_bands_to_filter": args.k_bands_to_filter,
        "attenuation_factor": args.attenuation_factor,
        "drop_rate": args.drop_rate,
        "filter_rate": args.filter_rate,
        "geo_risk_weight": args.geo_risk_weight,
        "spectral_risk_weight": args.spectral_risk_weight,
        "grad_smooth_step": args.grad_smooth_step,
        "grad_smooth_eps": args.grad_smooth_eps,
    })

    logger.info(f"{args.attacker} {args.model} {args.dataset}")
    logger.info(f"run_grad_wrt_wavelet_filter_new def result: {def_acc}")
