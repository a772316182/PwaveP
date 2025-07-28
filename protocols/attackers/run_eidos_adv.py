import argparse
import os
import sys

import torch
import torch_geometric
from loguru import logger
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
    from src.attackers.eidos.starter import start_eidos_attack
    from src.models.evaluator import validate, validate_adv_examples
    from src.utils.save_utils import pre_check_path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="pointnet",
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
    parser.add_argument(
        "--num_batches_to_attack",
        type=int,
        default=17,
        help="number of epochs to attack",
    )
    args = parser.parse_args()

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

    all_attacked_pc = start_eidos_attack(
        model=model,
        device=device,
        loader_train=loader_train,
        loader_test=loader_test,
        args=args,
        num_batches_to_attack=args.num_batches_to_attack,
    )

    clean_acc, asr = validate_adv_examples(all_attacked_pc, model, device)

    logger.info(
        f"hit adv at model {args.model} and dataset {args.dataset} attack result:"
    )
    logger.info(f"clean acc: {clean_acc['acc']}")
    logger.info(f"adv acc: {asr['acc']}")

    torch.save(
        all_attacked_pc,
        pre_check_path(
            f"{project_root_path}/attacked_data/eidos/{args.dataset.lower()}_{args.model.lower()}_res.pt"
        ),
    )
