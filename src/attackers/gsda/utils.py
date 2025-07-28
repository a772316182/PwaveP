from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from pytorch3d.ops import knn_points, knn_gather
from torch import nn
from torch.autograd import Variable


def _forward_step(
    net,
    pc_ori,
    input_curr_iter,
    normal_ori,
    ori_kappa,
    target,
    scale_const,
    targeted,
    v,
    num_classes,
    cls_loss_type="CE",
    dis_loss_type="CD",
    is_cd_single_side=False,
    confidence=0,
    dis_loss_weight=1,
    curv_loss_weight=1,
    curv_loss_knn=16,
    hd_loss_weight=0.1,
):
    device = pc_ori.device
    b, _, n = input_curr_iter.size()
    output_curr_iter = net(input_curr_iter)['logit']

    if cls_loss_type == "Margin":
        target_onehot = torch.zeros(target.size() + (num_classes,)).to(device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        fake = (target_onehot * output_curr_iter).sum(1)
        other = (
            (1.0 - target_onehot) * output_curr_iter - target_onehot * 10000.0
        ).max(1)[0]

        if targeted:
            # if targeted, optimize for making the other class most likely
            cls_loss = torch.clamp(
                other - fake + confidence, min=0.0
            )  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            cls_loss = torch.clamp(
                fake - other + confidence, min=0.0
            )  # equiv to max(..., 0.)

    elif cls_loss_type == "CE":
        if targeted:
            cls_loss = nn.CrossEntropyLoss(reduction="none").to(device)(
                output_curr_iter, Variable(target, requires_grad=False)
            )
        else:
            cls_loss = -nn.CrossEntropyLoss(reduction="none").to(device)(
                output_curr_iter, Variable(target, requires_grad=False)
            )
    elif cls_loss_type == "None":
        cls_loss = torch.FloatTensor(b).zero_().to(device)
    else:
        assert False, "Not support such clssification loss"

    info = "cls_loss: {0:6.4f}\t".format(cls_loss.mean().item())

    if dis_loss_type == "CD":
        if is_cd_single_side:
            dis_loss = pseudo_chamfer_loss(input_curr_iter, pc_ori)
        else:
            dis_loss = chamfer_loss(input_curr_iter, pc_ori)

        constrain_loss = dis_loss_weight * dis_loss
        info = info + "cd_loss: {0:6.4f}\t".format(dis_loss.mean().item())
    elif dis_loss_type == "L2":
        assert hd_loss_weight == 0
        dis_loss = norm_l2_loss(input_curr_iter, pc_ori)
        constrain_loss = dis_loss_weight * dis_loss
        info = info + "l2_loss: {0:6.4f}\t".format(dis_loss.mean().item())
    elif dis_loss_type == "None":
        dis_loss = 0
        constrain_loss = 0
    elif dis_loss_type == "Spectral":
        dis_loss = spectral_loss(input_curr_iter, pc_ori, v)
        constrain_loss = dis_loss_weight * dis_loss
        info = info + "spectral_loss: {0:6.4f}\t".format(dis_loss.mean().item())
    else:
        assert False, "Not support such distance loss"

    # hd_loss
    if hd_loss_weight != 0:
        hd_loss = hausdorff_loss(input_curr_iter, pc_ori)
        constrain_loss = constrain_loss + hd_loss_weight * hd_loss
        info = info + "hd_loss : {0:6.4f}\t".format(hd_loss.mean().item())
    else:
        hd_loss = 0

    # nor loss
    if curv_loss_weight != 0:
        adv_kappa, normal_curr_iter = _get_kappa_adv(
            input_curr_iter, pc_ori, normal_ori, curv_loss_knn
        )
        curv_loss = curvature_loss(input_curr_iter, pc_ori, adv_kappa, ori_kappa)
        constrain_loss = constrain_loss + curv_loss_weight * curv_loss
        info = info + "curv_loss : {0:6.4f}\t".format(curv_loss.mean().item())
    else:
        normal_curr_iter = torch.zeros(b, 3, n).to(device)
        curv_loss = 0

    scale_const = scale_const.float().to(device)
    loss_n = cls_loss + scale_const * constrain_loss
    loss = loss_n.mean()

    return (
        output_curr_iter,
        normal_curr_iter,
        loss,
        loss_n,
        cls_loss,
        dis_loss,
        hd_loss,
        curv_loss,
        constrain_loss,
        info,
    )


def _compare(output, target, gt, targeted):
    if targeted:
        return output == target
    else:
        return output != gt


def farthest_points_sample(obj_points, num_points):
    assert obj_points.size(1) == 3
    b, _, n = obj_points.size()

    device = obj_points.device

    selected = torch.randint(obj_points.size(2), [obj_points.size(0), 1]).to(device)
    dists = torch.full(
        [obj_points.size(0), obj_points.size(2)], fill_value=np.inf
    ).to(device)

    for _ in range(num_points - 1):
        dists = torch.min(
            dists,
            torch.norm(
                obj_points
                - torch.gather(
                    obj_points,
                    2,
                    selected[:, -1].unsqueeze(1).unsqueeze(2).expand(b, 3, 1),
                ),
                dim=1,
            ),
        )
        selected = torch.cat(
            [selected, torch.argmax(dists, dim=1, keepdim=True)], dim=1
        )
    res_points = torch.gather(
        obj_points, 2, selected.unsqueeze(1).expand(b, 3, num_points)
    )

    return res_points


def _normalize(input, p=2, dim=1, eps=1e-12):
    return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)


def norm_l2_loss(adv_pc, ori_pc):
    return ((adv_pc - ori_pc) ** 2).sum(1).sum(1)


def chamfer_loss(adv_pc, ori_pc):
    # Chamfer distance (two sides)
    # intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    # dis_loss = intra_dis.min(2)[0].mean(1) + intra_dis.min(1)[0].mean(1)
    adv_KNN = knn_points(
        adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1
    )  # [dists:[b,n,1], idx:[b,n,1]]
    ori_KNN = knn_points(
        ori_pc.permute(0, 2, 1), adv_pc.permute(0, 2, 1), K=1
    )  # [dists:[b,n,1], idx:[b,n,1]]
    dis_loss = adv_KNN.dists.contiguous().squeeze(-1).mean(
        -1
    ) + ori_KNN.dists.contiguous().squeeze(-1).mean(
        -1
    )  # [b]
    return dis_loss


def spectral_loss(adv_pc, ori_pc, v):
    x_ = torch.einsum(
        "bij,bjk->bik", v.transpose(1, 2), ori_pc.transpose(1, 2)
    )  # (b,n,3)
    x_adv = torch.einsum(
        "bij,bjk->bik", v.transpose(1, 2), adv_pc.transpose(1, 2)
    )  # (b,n,3)
    dis_loss = torch.square(x_adv - x_).sum(-1).sum(-1)  # [b]
    return dis_loss


def pseudo_chamfer_loss(adv_pc, ori_pc):
    # Chamfer pseudo distance (one side)
    # intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1) #b*n*n
    # dis_loss = intra_dis.min(2)[0].mean(1)
    adv_KNN = knn_points(
        adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1
    )  # [dists:[b,n,1], idx:[b,n,1]]
    dis_loss = adv_KNN.dists.contiguous().squeeze(-1).mean(-1)  # [b]
    return dis_loss


def hausdorff_loss(adv_pc, ori_pc):
    # dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    # hd_loss = torch.max(torch.min(dis, dim=2)[0], dim=1)[0]
    adv_KNN = knn_points(
        adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1
    )  # [dists:[b,n,1], idx:[b,n,1]]
    hd_loss = adv_KNN.dists.contiguous().squeeze(-1).max(-1)[0]  # [b]
    return hd_loss


def _get_kappa_ori(pc, normal, k=2):
    b, _, n = pc.size()
    # inter_dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
    # inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    # nn_pts = torch.gather(pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    inter_KNN = knn_points(
        pc.permute(0, 2, 1), pc.permute(0, 2, 1), K=k + 1
    )  # [dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = (
        knn_gather(pc.permute(0, 2, 1), inter_KNN.idx)
        .permute(0, 3, 1, 2)[:, :, :, 1:]
        .contiguous()
    )  # [b, 3, n ,k]
    vectors = nn_pts - pc.unsqueeze(3)
    vectors = _normalize(vectors)

    return torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)  # [b, n]


def _get_kappa_adv(adv_pc, ori_pc, ori_normal, k=2):
    b, _, n = adv_pc.size()
    # compute knn between advPC and oriPC to get normal n_p
    # intra_dis = ((adv_pc.unsqueeze(3) - ori_pc.unsqueeze(2))**2).sum(1)
    # intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
    # normal = torch.gather(ori_normal, 2, intra_idx.view(b,1,n).expand(b,3,n))
    intra_KNN = knn_points(
        adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1
    )  # [dists:[b,n,1], idx:[b,n,1]]
    normal = (
        knn_gather(ori_normal.permute(0, 2, 1), intra_KNN.idx)
        .permute(0, 3, 1, 2)
        .squeeze(3)
        .contiguous()
    )  # [b, 3, n]

    # compute knn between advPC and itself to get \|q-p\|_2
    # inter_dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1)
    # inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
    # nn_pts = torch.gather(adv_pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
    inter_KNN = knn_points(
        adv_pc.permute(0, 2, 1), adv_pc.permute(0, 2, 1), K=k + 1
    )  # [dists:[b,n,k+1], idx:[b,n,k+1]]
    nn_pts = (
        knn_gather(adv_pc.permute(0, 2, 1), inter_KNN.idx)
        .permute(0, 3, 1, 2)[:, :, :, 1:]
        .contiguous()
    )  # [b, 3, n ,k]
    vectors = nn_pts - adv_pc.unsqueeze(3)
    vectors = _normalize(vectors)

    return (
        torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2),
        normal,
    )  # [b, n], [b, 3, n]


def curvature_loss(adv_pc, ori_pc, adv_kappa, ori_kappa, k=2):
    b, _, n = adv_pc.size()

    # intra_dis = ((input_curr_iter.unsqueeze(3) - pc_ori.unsqueeze(2))**2).sum(1)
    # intra_idx = torch.topk(intra_dis, 1, dim=2, largest=False, sorted=True)[1]
    # knn_theta_normal = torch.gather(theta_normal, 1, intra_idx.view(b,n).expand(b,n))
    # curv_loss = ((curv_loss - knn_theta_normal)**2).mean(-1)

    intra_KNN = knn_points(
        adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1
    )  # [dists:[b,n,1], idx:[b,n,1]]
    onenn_ori_kappa = torch.gather(
        ori_kappa, 1, intra_KNN.idx.squeeze(-1)
    ).contiguous()  # [b, n]

    curv_loss = ((adv_kappa - onenn_ori_kappa) ** 2).mean(-1)

    return curv_loss


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
    b, _, n = adv_pc.size()
    # dis = ((adv_pc.unsqueeze(3) - adv_pc.unsqueeze(2))**2).sum(1) #[b,n,n]
    # dis, idx = torch.topk(dis, k+1, dim=2, largest=False, sorted=True)#[b,n,k+1]
    inter_KNN = knn_points(
        adv_pc.permute(0, 2, 1), adv_pc.permute(0, 2, 1), K=k + 1
    )  # [dists:[b,n,k+1], idx:[b,n,k+1]]

    knn_dis = inter_KNN.dists[:, :, 1:].contiguous().mean(-1)  # [b,n]
    knn_dis_mean = knn_dis.mean(-1)  # [b]
    knn_dis_std = knn_dis.std(-1)  # [b]
    threshold = knn_dis_mean + threshold_coef * knn_dis_std  # [b]

    condition = torch.gt(knn_dis, threshold.unsqueeze(1)).float()  # [b,n]
    dis_mean = knn_dis * condition  # [b,n]

    return dis_mean.mean(1)  # [b]
