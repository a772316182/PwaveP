from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch import nn
from tqdm import tqdm

from .bp_utils import boundary_projection_3
from .clip_utils import ClipPointsLinf
from .loss_utils import CWLoss
from .siadv_utils import (
    get_transformed_point_cloud,
    get_original_point_cloud,
    get_normal_vector,
)


class eidos_attack(nn.Module):
    def __init__(
        self,
        eps: float,
        step_size: float,
        max_steps: int,
        classifier: nn.Module,
        pre_head: Optional[nn.Module],
        num_class: int,
        top5_attack: bool,
        # boundary projection arguments
        bp: str,
        l2_weight: float,
        hd_weight: float,
        cd_weight: float,
        curv_weight: float,
        curv_loss_knn: int,
        stage2_steps: float,
        exponential_step: bool,
    ):
        super().__init__()
        self.eps = eps
        self.step_size = step_size
        self.max_steps = max_steps
        self.classifier = classifier
        self.pre_head = pre_head
        self.num_class = num_class
        self.top5_attack = top5_attack
        self.bp = bp
        self.si_grad_required = False

        self.boundary_projection = boundary_projection_3(
            l2_weight,
            hd_weight,
            cd_weight,
            curv_weight,
            curv_loss_knn,
            step_size,
            stage2_steps,
            max_steps,
            exponential_step,
        )

    def forward(self, points, target):
        """White-box I-FGSM with boundary projection based on shape-invariant sensitivity maps.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [B, N, 6].
            target (torch.cuda.LongTensor): the label for points, [B].
        """
        normal_vec = points[:, :, -3:].detach()  # N, [1, N, 3]
        normal_vec = normal_vec / torch.sqrt(
            torch.sum(normal_vec**2, dim=-1, keepdim=True)
        )  # N, [1, N, 3]
        points = points[:, :, :3].detach()  # P, [1, N, 3]
        ori_points = points.detach()
        clip_func = ClipPointsLinf(budget=self.eps)  # * np.sqrt(3*1024))
        stage2 = False

        self.boundary_projection.reset(points.size(0), device=points.device)

        for i in tqdm(range(self.max_steps)):

            if not stage2:
                # P -> P', detach()
                new_points, spin_axis_matrix, translation_matrix = (
                    get_transformed_point_cloud(points, normal_vec)
                )
                new_points = new_points.detach()
                new_points.requires_grad = True
                # P' -> P
                points = get_original_point_cloud(
                    new_points, spin_axis_matrix, translation_matrix
                )
                points = points.transpose(1, 2)  # P, [1, 3, N]
                # get white-box gradients
                logits = self.classifier.forward(self.pre_head(points))['logit']
                loss = CWLoss(
                    logits, target, kappa=0.0, tar=False, num_classes=self.num_class
                )
                self.classifier.zero_grad()
                loss.backward()
                grad = new_points.grad.data  # g, [1, N, 3]
                grad[:, :, 2] = 0.0
                # update P', P and N
                # # Linf
                # new_points = new_points - self.step_size * torch.sign(grad)

                # L2
                norm = torch.sum(grad**2, dim=[1, 2]) ** 0.5
                new_points = new_points - self.step_size * np.sqrt(3 * 1024) * grad / (
                    norm[:, None, None] + 1e-9
                )
                points = get_original_point_cloud(
                    # P, [1, N, 3]
                    new_points,
                    spin_axis_matrix,
                    translation_matrix,
                )
                points = clip_func(points, ori_points)

                points = points.detach()

                normal_vec = get_normal_vector(points)  # N, [1, N, 3]

                logits = self.classifier.forward(self.pre_head(points.transpose(1, 2)))['logit']

                logits = logits.argmax(-1)
                stage2 = (logits != target).all()

            else:
                points = points.detach()
                points.requires_grad = True

                logits = self.classifier.forward(self.pre_head(points.transpose(1, 2)))['logit']

                loss = (
                    logits.log_softmax(dim=-1)
                    .gather(dim=1, index=target.unsqueeze(1))
                    .sum()
                )
                self.classifier.zero_grad()
                loss.backward()

                g = points.grad.detach()

                g_norm = (g**2).sum((1, 2)).sqrt()
                g_norm.clamp_(min=1e-12)
                g_hat = g / g_norm[:, None, None]

                points = self.boundary_projection(
                    points, ori_points, normal_vec, g_hat, logits, target
                )

                normal_vec = get_normal_vector(points)

        with torch.no_grad():
            if self.boundary_projection.output_points is not None:
                adv_points = self.boundary_projection.output_points.clone()
            else:
                adv_points = points.clone()

            adv_logits = self.classifier.forward(self.pre_head(adv_points.transpose(1, 2)))['logit']
            adv_target = adv_logits.argmax(-1)

        if self.top5_attack:
            target_top_5 = adv_logits.topk(5)[1]
            if target in target_top_5:
                adv_target = target
            else:
                adv_target = -1

        del normal_vec, grad, new_points, spin_axis_matrix, translation_matrix
        asr = (adv_logits.data.max(1)[1] != target).sum().item()
        logger.info(f"ASR: {asr}/{len(points)}")
        return (
            adv_points,
            adv_target,
            asr,
        )


def build_eidos_attack(model, num_classes):
    return eidos_attack(
        eps=0.16,
        step_size=0.007,
        max_steps=100,
        classifier=model,
        pre_head=torch.nn.Identity(),
        num_class=num_classes,
        top5_attack=False,
        # boundary projection arguments
        bp="bp3",
        l2_weight=1.0,
        hd_weight=1.0,
        cd_weight=1.0,
        curv_weight=1.0,
        curv_loss_knn=16,
        stage2_steps=0.030,
        exponential_step=True,
    )
