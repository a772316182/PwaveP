from __future__ import absolute_import, division, print_function

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_gather, knn_points


def _normalize(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


class loss_wrapper(nn.Module):
    def __init__(
        self,
        loss_func: Callable[..., torch.Tensor],
        channel_first: bool = True,
        keep_batch: bool = False,
        need_normal: bool = False,
    ):
        super().__init__()
        self.channel_first = channel_first
        self.keep_batch = keep_batch
        self.loss_func = loss_func
        self.need_normal = need_normal

    def forward(
        self, adv_pc: torch.Tensor, ori_pc: torch.Tensor, normal_pc: torch.Tensor = None
    ):
        """
        Args:
            adv_pc: [B, 3, N] if self.channel_first else [B, N, 3]
            ori_pc: [B, 3, N] if self.channel_first else [B, N, 3]
            normal_pc: [B, 3, N] if self.channel_first else [B, N, 3]

        Returns:
            scalar if self.keep_dim else [B]
        """
        if self.channel_first:
            adv_pc = adv_pc.transpose(1, 2)
            ori_pc = ori_pc.transpose(1, 2)

        if self.need_normal:
            loss: torch.Tensor = self.loss_func(adv_pc, ori_pc, normal_pc)
        else:
            loss: torch.Tensor = self.loss_func(adv_pc, ori_pc)

        if not self.keep_batch:
            return loss.sum()

        return loss


def norm_l2_loss(adv_pc: torch.Tensor, ori_pc: torch.Tensor):
    """
    Args:
        adv_pc: [B, N, 3] adversarial point cloud
        ori_pc: [B, N, 3] original point cloud

    Returns:
        [B] L2 loss per sample
    """
    return ((adv_pc - ori_pc) ** 2).sum((1, 2))


def chamfer_loss(adv_pc, ori_pc):
    """
    Args:
        adv_pc: [B, N, 3] adversarial point cloud
        ori_pc: [B, N, 3] original point cloud

    Returns:
        [B] Symmetric chamfer loss per sample
    """
    # Chamfer distance (two sides)
    adv_KNN = knn_points(adv_pc, ori_pc, K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    ori_KNN = knn_points(ori_pc, adv_pc, K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    dis_loss = adv_KNN.dists.contiguous().squeeze(-1).mean(
        -1
    ) + ori_KNN.dists.contiguous().squeeze(-1).mean(
        -1
    )  # [b]
    return dis_loss


def pseudo_chamfer_loss(adv_pc, ori_pc):
    # Chamfer pseudo distance (one side)

    adv_KNN = knn_points(adv_pc, ori_pc, K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    dis_loss = adv_KNN.dists.contiguous().squeeze(-1).mean(-1)  # [b]
    return dis_loss


def hausdorff_loss(adv_pc, ori_pc):
    """
    Args:
        adv_pc: [B, N, 3] adversarial point cloud
        ori_pc: [B, N, 3] original point cloud

    Returns:
        [B] Hausdorff loss per sample
    """
    adv_KNN = knn_points(adv_pc, ori_pc, K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    hd_loss = adv_KNN.dists.contiguous().squeeze(-1).max(-1)[0]  # [b]
    return hd_loss


def _get_kappa_ori(pc, normal, k=2):
    b, n, _ = pc.size()
    inter_KNN = knn_points(pc, pc, K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = (
        knn_gather(pc, inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :, 1:].contiguous()
    )  # [b, 3, n, k]
    vectors = nn_pts - pc.transpose(1, 2).unsqueeze(3)
    vectors = _normalize(vectors)

    return torch.abs((vectors * normal.transpose(1, 2).unsqueeze(3)).sum(1)).mean(2)


def _get_kappa_adv(adv_pc, ori_pc, ori_normal, k=2):
    b, n, _ = adv_pc.size()

    intra_KNN = knn_points(adv_pc, ori_pc, K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    normal = (
        knn_gather(ori_normal, intra_KNN.idx)
        .permute(0, 3, 1, 2)
        .squeeze(3)
        .contiguous()
    )  # [b, 3, n]

    # compute knn between advPC and itself to get \|q-p\|_2
    inter_KNN = knn_points(adv_pc, adv_pc, K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = (
        knn_gather(adv_pc, inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :, 1:].contiguous()
    )  # [b, 3, n ,k]
    vectors = nn_pts - adv_pc.transpose(1, 2).unsqueeze(3)
    vectors = _normalize(vectors)

    return (
        torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2),
        normal,
    )  # [b, n], [b, n, 3]


def curvature_loss(adv_pc, ori_pc, adv_kappa, ori_kappa, k=2):
    b, n, _ = adv_pc.size()

    intra_KNN = knn_points(adv_pc, ori_pc, K=1)  # [dists:[b,n,1], idx:[b,n,1]]
    onenn_ori_kappa = torch.gather(
        ori_kappa, 1, intra_KNN.idx.squeeze(-1)
    ).contiguous()  # [b, n]

    curv_loss = ((adv_kappa - onenn_ori_kappa) ** 2).mean(-1)

    return curv_loss


def local_curvature_loss(adv_pc, ori_pc, ori_normal, k=2):
    ori_kappa = _get_kappa_ori(ori_pc, ori_normal, k)
    adv_kappa, _ = _get_kappa_adv(adv_pc, ori_pc, ori_normal, k)
    local_curv_loss = curvature_loss(adv_pc, ori_pc, adv_kappa, ori_kappa)

    return local_curv_loss


def direction_loss(adv_pc, ori_pc):
    # dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    # hd_loss = torch.max(torch.min(dis, dim=2)[0], dim=1)[0]
    adv_KNN = knn_points(
        adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1
    )  # [dists:[b,n,1], idx:[b,n,1]]
    ori_idx = torch.arange(adv_KNN.idx.size(1)).cuda().view(-1, 1)

    ori_pc_tr = ori_pc.permute(0, 2, 1)
    adv_pc_tr = adv_pc.permute(0, 2, 1)
    knn_pc_tr = knn_gather(ori_pc_tr, adv_KNN.idx).view(adv_pc_tr.size())

    p1 = ori_pc_tr - adv_pc_tr
    p2 = knn_pc_tr - adv_pc_tr
    dir_loss = F.cosine_similarity(p1, p2, dim=(2))
    return (dir_loss < 0.0).sum()


def direction_loss(adv_pc, ori_pc):
    # dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    # hd_loss = torch.max(torch.min(dis, dim=2)[0], dim=1)[0]
    adv_KNN = knn_points(
        adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1
    )  # [dists:[b,n,1], idx:[b,n,1]]
    ori_idx = torch.arange(adv_KNN.idx.size(1)).cuda().view(-1, 1)

    ori_pc_tr = ori_pc.permute(0, 2, 1)
    adv_pc_tr = adv_pc.permute(0, 2, 1)
    knn_pc_tr = knn_gather(ori_pc_tr, adv_KNN.idx).view(adv_pc_tr.size())

    p1 = (ori_pc_tr - adv_pc_tr).view(adv_pc_tr.size(0), -1)
    p2 = (knn_pc_tr - adv_pc_tr).view(adv_pc_tr.size(0), -1)
    dir_loss = F.cosine_similarity(p1, p2, dim=(1))
    return (dir_loss < 0.0).sum()


def hausdorff_transform_loss(adv_pc, ori_pc):
    # dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    # hd_loss = torch.max(torch.min(dis, dim=2)[0], dim=1)[0]
    adv_KNN = knn_points(
        adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1
    )  # [dists:[b,n,1], idx:[b,n,1]]
    ori_idx = torch.arange(adv_KNN.idx.size(1)).cuda().view(-1, 1)
    hd_transform_loss = (adv_KNN.idx != ori_idx[np.newaxis, :, :]).sum((1, 2))

    return hd_transform_loss


def displacement_loss(adv_pc, ori_pc, k=16):
    b, _, n = adv_pc.size()
    with torch.no_grad():
        inter_dis = ((ori_pc.unsqueeze(3) - ori_pc.unsqueeze(2)) ** 2).sum(1)
        inter_idx = torch.topk(inter_dis, k + 1, dim=2, largest=False, sorted=True)[1][
            :, :, 1:
        ].contiguous()

    theta_distance = ((adv_pc - ori_pc) ** 2).sum(1)
    nn_theta_distances = torch.gather(theta_distance, 1, inter_idx.view(b, n * k)).view(
        b, n, k
    )
    return ((nn_theta_distances - theta_distance.unsqueeze(2)) ** 2).mean(2)


def corresponding_normal_loss(adv_pc, normal, k=2):
    b, _, n = adv_pc.size()

    inter_dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2)) ** 2).sum(1)
    inter_idx = torch.topk(inter_dis, k + 1, dim=2, largest=False, sorted=True)[1][
        :, :, 1:
    ].contiguous()
    nn_pts = torch.gather(
        adv_pc, 2, inter_idx.view(b, 1, n * k).expand(b, 3, n * k)
    ).view(b, 3, n, k)
    vectors = nn_pts - adv_pc.unsqueeze(3)
    vectors = _normalize(vectors)
    return torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)


def repulsion_loss(pc, k=4, h=0.03):
    dis = ((pc.unsqueeze(3) - pc.unsqueeze(2)) ** 2).sum(1)
    dis = torch.topk(dis, k + 1, dim=2, largest=False, sorted=True)[0][
        :, :, 1:
    ].contiguous()

    return -(dis * torch.exp(-(dis**2) / (h**2))).mean(2)


def distance_kmean_loss(pc, k):
    b, _, n = pc.size()
    dis = ((pc.unsqueeze(3) - pc.unsqueeze(2) + 1e-12) ** 2).sum(1).sqrt()
    dis, idx = torch.topk(dis, k + 1, dim=2, largest=False, sorted=True)
    dis_mean = dis[:, :, 1:].contiguous().mean(-1)  # b*n
    idx = idx[:, :, 1:].contiguous()
    dis_mean_k = torch.gather(dis_mean, 1, idx.view(b, n * k)).view(b, n, k)

    return torch.abs(dis_mean.unsqueeze(2) - dis_mean_k).mean(-1)


def kNN_smoothing_loss(adv_pc, k, threshold_coef=1.05):
    b, n, _ = adv_pc.size()
    inter_KNN = knn_points(adv_pc, adv_pc, K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]

    knn_dis = inter_KNN.dists[:, :, 1:].contiguous().mean(-1)  # [b,n]
    knn_dis_mean = knn_dis.mean(-1)  # [b]
    knn_dis_std = knn_dis.std(-1)  # [b]
    threshold = knn_dis_mean + threshold_coef * knn_dis_std  # [b]

    condition = torch.gt(knn_dis, threshold.unsqueeze(1)).float()  # [b,n]
    dis_mean = knn_dis * condition  # [b,n]

    return dis_mean.mean(1)  # [b]


def spectral_loss(adv_pc, ori_pc, v):
    x_ = torch.einsum(
        "bij,bjk->bik", v.transpose(1, 2), ori_pc.transpose(1, 2)
    )  # (b,n,3)
    x_adv = torch.einsum(
        "bij,bjk->bik", v.transpose(1, 2), adv_pc.transpose(1, 2)
    )  # (b,n,3)
    dis_loss = torch.square(x_adv - x_).sum(-1).sum(-1)  # [b]
    return dis_loss


def CWLoss(
    logits, target, kappa=0, tar=False, num_classes=40, top5_attack: bool = False
):
    """Carlini & Wagner attack loss.

    Args:
        logits (torch.cuda.FloatTensor): the predicted logits, [B, num_classes].
        target (torch.cuda.LongTensor): the label for points, [B].
    """
    target_one_hot = torch.nn.functional.one_hot(target, num_classes).float()

    real = torch.sum(target_one_hot * logits, 1)
    if not top5_attack:
        # top-1 attack
        other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10000), 1)[
            0
        ]
    else:
        # top-5 attack
        other = torch.topk((1 - target_one_hot) * logits - (target_one_hot * 10000), 5)[
            0
        ][:, 4]
    kappa = torch.zeros_like(other).fill_(kappa)

    if tar:
        return torch.sum(torch.max(other - real, kappa))
    else:
        return torch.sum(torch.max(real - other, kappa))


__all_loss__ = {
    "l2": norm_l2_loss,
    "hd": hausdorff_loss,
    "cd": pseudo_chamfer_loss,
    "curv": local_curvature_loss,
}
