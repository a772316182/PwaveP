import pytorch3d.loss
import torch
from loguru import logger


@torch.no_grad()
def eval_cd(
    ori_pc: torch.Tensor, adv_pc: torch.Tensor, purified_pc: torch.Tensor, need_cd=True
):
    ori_pc, adv_pc, purified_pc = (
        ori_pc.float().detach().clone(),
        adv_pc.float().detach().clone(),
        purified_pc.float().detach().clone(),
    )
    cd_clean_adv = pytorch3d.loss.chamfer_distance(ori_pc, adv_pc)
    cd_clean_def = pytorch3d.loss.chamfer_distance(ori_pc, purified_pc)

    logger.warning("cd_clean_adv")
    logger.warning(cd_clean_adv[0].item())
    logger.warning("cd_clean_def")
    logger.warning(cd_clean_def[0].item())

    if need_cd:
        return cd_clean_adv[0].item(), cd_clean_def[0].item()
    return None
