from __future__ import absolute_import, division, print_function

import os

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from pytorch3d.ops import knn_points, knn_gather
from torch.autograd import Variable
from tqdm import tqdm

from .spectral_attack import eig_vector
from .utils import (
    _compare,
    farthest_points_sample,
    norm_l2_loss,
    chamfer_loss,
    pseudo_chamfer_loss,
    hausdorff_loss,
    curvature_loss,
    _get_kappa_ori,
    _get_kappa_adv,
    spectral_loss,
)


class gsda_attack(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        defense_head: nn.Module,
        attack_label: str,
        curv_loss_weight: float,
        curv_loss_knn: int,
        initial_const: float,
        iter_max_steps: int,
        cls_loss_type: str,
        dis_loss_type: str,
        hd_loss_weight: float,
        binary_max_steps: int,
        optim: str,
        dis_loss_weight: float,
        lr: float,
        is_use_lr_scheduler: bool,
        cc_linf: float,
        npoint: int,
        confidence: int,
        classes: int,
        eval_num: int,
        uniform_loss_weight: float,
        band_frequency: list[int],
        spectral_attack: bool,
        spectral_offset: bool,
        KNN: int,
        is_partial_var: bool,
        is_subsample_opt: bool,
        spectral_restrict: float,
        is_cd_single_side: bool,
    ):
        super().__init__()
        self.model = model
        self.defense_head = defense_head
        self.attack_label = attack_label
        self.curv_loss_weight = curv_loss_weight
        self.curv_loss_knn = curv_loss_knn
        self.initial_const = initial_const
        self.iter_max_steps = iter_max_steps
        self.cls_loss_type = cls_loss_type
        self.dis_loss_type = dis_loss_type
        self.lr = lr
        self.hd_loss_weight = hd_loss_weight
        self.binary_max_steps = binary_max_steps
        self.optim = optim
        self.dis_loss_weight = dis_loss_weight
        self.is_use_lr_scheduler = is_use_lr_scheduler
        self.cc_linf = cc_linf
        self.npoint = npoint
        self.confidence = confidence
        self.classes = classes
        self.eval_num = eval_num
        self.uniform_loss_weight = uniform_loss_weight
        self.band_frequency = band_frequency
        self.spectral_attack = spectral_attack
        self.spectral_offset = spectral_offset
        self.KNN = KNN
        self.is_partial_var = is_partial_var
        self.is_subsample_opt = is_subsample_opt
        self.spectral_restrict = spectral_restrict
        self.is_cd_single_side = is_cd_single_side

        self.output_path = None

    def GFT(self, pc_ori, K, factor):
        x = pc_ori.transpose(0, 1)  # (b,n,3)
        b, n, _ = x.shape
        v = eig_vector(x, K)
        x_ = torch.einsum("bij,bjk->bik", v.transpose(1, 2), x)  # (b,n,3)
        x_ = torch.einsum("bij,bi->bij", x_, factor)
        x = torch.einsum("bij,bjk->bik", v, x_)
        return x

    def resample_reconstruct_from_pc(
        self, output_file_name, pc, normal=None, reconstruct_type="PRS"
    ):
        assert pc.size() == 2
        assert pc.size(2) == 3
        assert normal.size() == pc.size()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        if normal:
            pcd.normals = o3d.utility.Vector3dVector(normal)

        if reconstruct_type == "BPA":
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 3 * avg_dist

            bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector([radius, radius * 2])
            )

            output_mesh = bpa_mesh.simplify_quadric_decimation(100000)
            output_mesh.remove_degenerate_triangles()
            output_mesh.remove_duplicated_triangles()
            output_mesh.remove_duplicated_vertices()
            output_mesh.remove_non_manifold_edges()
        elif reconstruct_type == "PRS":
            poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8, width=0, scale=1.1, linear_fit=False
            )[0]
            bbox = pcd.get_axis_aligned_bounding_box()
            output_mesh = poisson_mesh.crop(bbox)

        o3d.io.write_triangle_mesh(
            os.path.join(self.output_path, output_file_name + "ply"), output_mesh
        )

        return o3d.geometry.TriangleMesh.sample_points_uniformly(
            output_mesh, number_of_points=self.npoint
        )

    def offset_proj(self, offset, ori_pc, ori_normal, project="dir"):
        # offset: shape [b, 3, n], perturbation offset of each point
        # normal: shape [b, 3, n], normal vector of the object
        device = ori_pc.device

        condition_inner = torch.zeros(offset.shape).to(device).byte()

        intra_KNN = knn_points(
            offset.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1
        )  # [dists:[b,n,1], idx:[b,n,1]]
        normal = (
            knn_gather(ori_normal.permute(0, 2, 1), intra_KNN.idx)
            .permute(0, 3, 1, 2)
            .squeeze(3)
            .contiguous()
        )  # [b, 3, n]

        normal_len = (normal**2).sum(1, keepdim=True).sqrt()
        normal_len_expand = normal_len.expand_as(offset)  # [b, 3, n]

        # add 1e-6 to avoid dividing by zero
        offset_projected = (
            (offset * normal / (normal_len_expand + 1e-6)).sum(1, keepdim=True)
            * normal
            / (normal_len_expand + 1e-6)
        )

        # let perturb be the projected ones
        offset = torch.where(condition_inner, offset, offset_projected)

        return offset

    def find_offset(self, ori_pc, adv_pc):
        intra_KNN = knn_points(
            adv_pc.permute(0, 2, 1), ori_pc.permute(0, 2, 1), K=1
        )  # [dists:[b,n,1], idx:[b,n,1]]
        knn_pc = (
            knn_gather(ori_pc.permute(0, 2, 1), intra_KNN.idx)
            .permute(0, 3, 1, 2)
            .squeeze(3)
            .contiguous()
        )  # [b, 3, n]

        real_offset = adv_pc - knn_pc

        return real_offset

    def lp_clip(self, offset, cc_linf):
        lengths = (offset**2).sum(1, keepdim=True).sqrt()  # [b, 1, n]
        lengths_expand = lengths.expand_as(offset)  # [b, 3, n]

        condition = lengths > 1e-6
        offset_scaled = torch.where(
            condition, offset / lengths_expand * cc_linf, torch.zeros_like(offset)
        )

        condition = lengths < cc_linf
        offset = torch.where(condition, offset, offset_scaled)

        return offset

    def _forward_step(
        self,
        pc_ori: torch.Tensor,
        input_curr_iter: torch.Tensor,
        normal_ori: torch.Tensor,
        ori_kappa: torch.Tensor,
        target,
        scale_const,
        targeted,
        v,
    ):
        device = pc_ori.device
        # needed self:[arch, classes, cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn]
        b, n, _ = input_curr_iter.size()

        output_curr_iter = self.model(
            self.defense_head(input_curr_iter.transpose(1, 2))
        )["logit"]

        if self.cls_loss_type == "Margin":
            target_onehot = torch.zeros(target.size() + (self.classes,)).to(device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

            fake = (target_onehot * output_curr_iter).sum(1)
            other = (
                (1.0 - target_onehot) * output_curr_iter - target_onehot * 10000.0
            ).max(1)[0]

            if targeted:
                # if targeted, optimize for making the other class most likely
                cls_loss = torch.clamp(
                    other - fake + self.confidence, min=0.0
                )  # equiv to max(..., 0.)
            else:
                # if non-targeted, optimize for making this class least likely.
                cls_loss = torch.clamp(
                    fake - other + self.confidence, min=0.0
                )  # equiv to max(..., 0.)

        elif self.cls_loss_type == "CE":
            if targeted:
                cls_loss = nn.CrossEntropyLoss(reduction="none").to(device)(
                    output_curr_iter, Variable(target, requires_grad=False)
                )
            else:
                cls_loss = -nn.CrossEntropyLoss(reduction="none").to(device)(
                    output_curr_iter, Variable(target, requires_grad=False)
                )
        elif self.cls_loss_type == "None":
            cls_loss = torch.FloatTensor(b).zero_().to(device)
        else:
            assert False, "Not support such clssification loss"

        info = "cls_loss: {0:6.4f}\t".format(cls_loss.mean().item())

        if self.dis_loss_type == "CD":
            if self.is_cd_single_side:
                dis_loss = pseudo_chamfer_loss(input_curr_iter, pc_ori)
            else:
                dis_loss = chamfer_loss(input_curr_iter, pc_ori)

            constrain_loss = self.dis_loss_weight * dis_loss
            info = info + "cd_loss: {0:6.4f}\t".format(dis_loss.mean().item())
        elif self.dis_loss_type == "L2":
            assert self.hd_loss_weight == 0
            dis_loss = norm_l2_loss(input_curr_iter, pc_ori)
            constrain_loss = self.dis_loss_weight * dis_loss
            info = info + "l2_loss: {0:6.4f}\t".format(dis_loss.mean().item())
        elif self.dis_loss_type == "None":
            dis_loss = 0
            constrain_loss = 0
        elif self.dis_loss_type == "Spectral":
            dis_loss = spectral_loss(input_curr_iter, pc_ori, v)
            constrain_loss = self.dis_loss_weight * dis_loss
            info = info + "spectral_loss: {0:6.4f}\t".format(dis_loss.mean().item())
        else:
            assert False, "Not support such distance loss"

        # hd_loss
        if self.hd_loss_weight != 0:
            hd_loss = hausdorff_loss(input_curr_iter, pc_ori)
            constrain_loss = constrain_loss + self.hd_loss_weight * hd_loss
            info = info + "hd_loss : {0:6.4f}\t".format(hd_loss.mean().item())
        else:
            hd_loss = 0

        # nor loss
        if self.curv_loss_weight != 0:
            adv_kappa, normal_curr_iter = _get_kappa_adv(
                input_curr_iter, pc_ori, normal_ori, self.curv_loss_knn
            )
            curv_loss = curvature_loss(input_curr_iter, pc_ori, adv_kappa, ori_kappa)
            constrain_loss = constrain_loss + self.curv_loss_weight * curv_loss
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

    def _forward_step_bp(
        self,
        net,
        defense_head,
        pc_ori,
        input_curr_iter,
        normal_ori,
        ori_kappa,
        target,
        scale_const,
        targeted,
        v,
    ):
        device = pc_ori.device
        # needed self:[arch, classes, cls_loss_type, confidence, dis_loss_type, is_cd_single_side, dis_loss_weight, hd_loss_weight, curv_loss_weight, curv_loss_knn]
        b, _, n = input_curr_iter.size()
        if not defense_head is None:
            output_curr_iter = net(defense_head(input_curr_iter))
        else:
            output_curr_iter = net(input_curr_iter)

        if self.cls_loss_type == "Margin":
            target_onehot = torch.zeros(target.size() + (self.classes,)).to(device)
            target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

            fake = (target_onehot * output_curr_iter).sum(1)
            other = (
                (1.0 - target_onehot) * output_curr_iter - target_onehot * 10000.0
            ).max(1)[0]

            if targeted:
                # if targeted, optimize for making the other class most likely
                cls_loss = torch.clamp(
                    other - fake + self.confidence, min=0.0
                )  # equiv to max(..., 0.)
            else:
                # if non-targeted, optimize for making this class least likely.
                cls_loss = torch.clamp(
                    fake - other + self.confidence, min=0.0
                )  # equiv to max(..., 0.)

        elif self.cls_loss_type == "CE":
            if targeted:
                cls_loss = nn.CrossEntropyLoss(reduction="none").to(device)(
                    output_curr_iter, Variable(target, requires_grad=False)
                )
            else:
                cls_loss = -nn.CrossEntropyLoss(reduction="none").to(device)(
                    output_curr_iter, Variable(target, requires_grad=False)
                )
        elif self.cls_loss_type == "None":
            cls_loss = torch.FloatTensor(b).zero_().to(device)
        else:
            assert False, "Not support such clssification loss"

        info = "cls_loss: {0:6.4f}\t".format(cls_loss.mean().item())

        loss_n = cls_loss
        loss = loss_n.mean()

        return (
            output_curr_iter,
            None,
            loss,
            loss_n,
            cls_loss,
            None,
            None,
            None,
            None,
            info,
        )

    def forward(self, points: torch.Tensor, target: torch.Tensor):
        device = points.device

        if self.attack_label == "Untarget":
            targeted = False
        else:
            targeted = True

        pc_ori = points[:, :, :3]
        normal_ori = points[:, :, 3:]
        gt_labels = target.view(-1)
        gt_target = gt_labels.view(-1)

        batch_size, num_points, _ = pc_ori.size()

        if self.attack_label == "Untarget":
            target = gt_target.to(device)
        else:
            target = target.view(-1)

        if self.curv_loss_weight != 0:
            kappa_ori = _get_kappa_ori(pc_ori, normal_ori, self.curv_loss_knn)
        else:
            kappa_ori = None

        lower_bound = torch.ones(batch_size) * 0
        scale_const = torch.ones(batch_size) * self.initial_const
        upper_bound = torch.ones(batch_size) * 1e10

        best_loss = [1e10] * batch_size
        best_attack = pc_ori.clone()
        best_x_ = torch.zeros(batch_size, num_points, 3).to(device)
        best_gft = torch.zeros(batch_size, num_points, 3).to(device)

        best_attack_step = [-1] * batch_size
        best_attack_BS_idx = [-1] * batch_size
        all_loss_list = [[-1] * batch_size] * self.iter_max_steps
        for search_step in tqdm(range(self.binary_max_steps)):
            iter_best_loss = [1e10] * batch_size
            iter_best_score = [-1] * batch_size
            constrain_loss = torch.ones(batch_size) * 1e10
            attack_success = torch.zeros(batch_size).to(device)

            input_all = None

            for step in range(self.iter_max_steps):
                if step == 0:
                    offset = torch.zeros(batch_size, num_points, 3).to(device)
                    nn.init.normal_(offset, mean=0, std=1e-3)
                    offset.requires_grad_()
                    factor = torch.zeros(batch_size, num_points, 3).to(device)
                    response = torch.zeros(batch_size, 5, 3).to(device)
                    response1 = torch.zeros(batch_size, num_points, 3).to(device)
                    response[:, 0, :] = 1
                    mask = torch.ones(batch_size, num_points, 3).to(device)
                    mask[:, : self.band_frequency[0], :] = 0
                    mask[:, self.band_frequency[1] :, :] = 0
                    if self.spectral_attack:
                        nn.init.normal_(factor, mean=0, std=1e-3)
                        nn.init.normal_(response, mean=0, std=1e-3)
                        factor.requires_grad_()
                        response.requires_grad_()
                        response1.requires_grad_()
                        if self.optim == "adam":
                            optimizer = optim.Adam(
                                [factor, response, response1], lr=self.lr
                            )
                        elif self.optim == "sgd":
                            optimizer = optim.SGD(
                                [factor, response, response1], lr=self.lr
                            )
                        else:
                            assert False, "Not support such optimizer."
                        x = pc_ori.clone()  # (batch_size,num_points,3)

                        K = self.KNN

                        v, laplacian, u = eig_vector(x, K)

                        u = u.unsqueeze(-1)
                        u_ = torch.cat(
                            (
                                torch.ones_like(u).to(u.device),
                                u,
                                u * u,
                                u * u * u,
                                u * u * u * u,
                            ),
                            dim=-1,
                        )  # (batch_size, num_points, 5)
                        x_ = torch.einsum(
                            "bij,bjk->bik", v.transpose(1, 2), x
                        )  # (batch_size,num_points,3)
                    else:
                        if self.optim == "adam":
                            optimizer = optim.Adam([offset], lr=self.lr)
                        elif self.optim == "sgd":
                            optimizer = optim.SGD([offset], lr=self.lr)
                        else:
                            assert False, "Not support such optimizer."

                    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                        optimizer, gamma=0.9990, last_epoch=-1
                    )

                    periodical_pc = pc_ori.clone()

                if self.spectral_attack:
                    gft = x_.clone()
                    if self.spectral_offset:
                        if self.spectral_restrict != 0:
                            factor_relative = torch.clamp(
                                factor / x_,
                                min=-self.spectral_restrict,
                                max=self.spectral_restrict,
                            )
                            factor_ = x_.mul(factor_relative)
                        else:
                            factor_ = factor
                        gft += factor_ * mask
                    GFT_pc = torch.einsum("bij,bjk->bik", v, gft)
                    input_all = GFT_pc
                else:
                    input_all = periodical_pc + offset

                if (
                    (input_all.size(1) > self.npoint)
                    and (not self.is_partial_var)
                    and self.is_subsample_opt
                ):
                    input_curr_iter = farthest_points_sample(input_all, self.npoint)
                else:
                    input_curr_iter = input_all

                with torch.no_grad():
                    for k in range(batch_size):
                        if input_curr_iter.size(1) < input_all.size(1):
                            batch_k_pc = farthest_points_sample(
                                torch.cat([input_all[k].unsqueeze(0)] * self.eval_num),
                                self.npoint,
                            )
                            batch_k_adv_output = self.model(
                                self.defense_head(batch_k_pc.transpose(1, 2))
                            )["logit"]
                            attack_success[k] = (
                                _compare(
                                    torch.max(batch_k_adv_output, 1)[1].data,
                                    target[k],
                                    gt_target[k],
                                    targeted,
                                ).sum()
                                > 0.5 * self.eval_num
                            )
                            output_label = (
                                torch.max(batch_k_adv_output, 1)[1].mode().values.item()
                            )
                        else:
                            adv_output = self.model(
                                self.defense_head(input_all[k : k + 1].transpose(1, 2))
                            )["logit"]
                            output_label = torch.argmax(adv_output).item()
                            attack_success[k] = _compare(
                                output_label, target[k], gt_target[k].to(device), targeted
                            ).item()

                        metric = constrain_loss[k].item()

                        if attack_success[k] and (metric < best_loss[k]):

                            best_loss[k] = metric
                            best_attack[k] = input_all.data[k].clone()
                            best_attack_BS_idx[k] = search_step
                            best_attack_step[k] = step
                            if self.spectral_attack:
                                best_gft[k] = gft[k].clone()
                                best_x_[k] = x_[k].clone()

                        if attack_success[k] and (metric < iter_best_loss[k]):
                            iter_best_loss[k] = metric
                            iter_best_score[k] = output_label

                (
                    _,
                    normal_curr_iter,
                    loss,
                    loss_n,
                    cls_loss,
                    dis_loss,
                    hd_loss,
                    nor_loss,
                    constrain_loss,
                    info,
                ) = self._forward_step(
                    pc_ori,
                    input_curr_iter,
                    normal_ori,
                    kappa_ori,
                    target,
                    scale_const,
                    targeted,
                    v,
                )

                print(info)

                all_loss_list[step] = loss_n.detach().tolist()

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()
                if self.is_use_lr_scheduler:
                    lr_scheduler.step()

                if self.cc_linf != 0:
                    with torch.no_grad():
                        proj_offset = self.lp_clip(offset, self.cc_linf)
                        offset.data = proj_offset.data

            # adjust the scale constants
            for k in range(batch_size):
                if (
                    _compare(
                        output_label, target[k], gt_target[k].to(device), targeted
                    ).item()
                    and iter_best_score[k] != -1
                ):
                    lower_bound[k] = max(lower_bound[k], scale_const[k])
                    if upper_bound[k] < 1e9:
                        scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5
                    else:
                        scale_const[k] *= 2
                else:
                    upper_bound[k] = min(upper_bound[k], scale_const[k])
                    if upper_bound[k] < 1e9:
                        scale_const[k] = (lower_bound[k] + upper_bound[k]) * 0.5

        return (best_attack, target, 0)


def build_gsda(model, num_classes):
    return gsda_attack(
        model=model,
        defense_head=torch.nn.Identity(),
        attack_label="Untarget",
        curv_loss_weight=1.0,
        curv_loss_knn=16,
        initial_const=10,
        iter_max_steps=500,
        cls_loss_type="CE",
        dis_loss_type="CD",
        hd_loss_weight=0.1,
        binary_max_steps=10,
        optim="adam",
        dis_loss_weight=1.0,
        lr=0.010,
        is_use_lr_scheduler=False,
        cc_linf=0.0,
        npoint=1024,
        confidence=0,
        classes=num_classes,
        eval_num=1,
        uniform_loss_weight=0.0,
        band_frequency=[0, 1024],
        spectral_attack=True,
        spectral_offset=True,
        KNN=10,
        is_partial_var=False,
        is_subsample_opt=False,
        spectral_restrict=0.0,
        is_cd_single_side=False,
    )
