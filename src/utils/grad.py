import numpy
import torch
from torch.nn.functional import cross_entropy


def extract_grad_on_all_classes(num_classes, simple_model, target_model_logits, device):
    grad_on_different_labels = []
    for j in range(num_classes):
        loss = cross_entropy(target_model_logits, torch.tensor([j], device=device))
        model_diag_weight_grad_clean = (
            torch.autograd.grad(
                loss,
                simple_model.diag_weight_stable,
                create_graph=True,
            )[0]
            .detach()
            .clone()
            .squeeze()
            .cpu()
            .numpy()
        )
        grad_on_different_labels.append(model_diag_weight_grad_clean)

    return numpy.stack(grad_on_different_labels)


def extract_grad_for_just_weight(simple_model, target_model_logits):
    pred_label = torch.argmax(target_model_logits, dim=-1)
    loss = cross_entropy(target_model_logits, pred_label)
    model_diag_weight_grad_clean = (
        torch.autograd.grad(
            loss,
            simple_model.weight,
            create_graph=True,
        )[0]
        .detach()
        .clone()
        .squeeze()
        .cpu()
        .numpy()
    )
    return model_diag_weight_grad_clean


def extract_grad_on_all_classes_for_just_weight(
    num_classes, simple_model, target_model_logits, device
):
    grad_on_different_labels = []
    for j in range(num_classes):
        loss = cross_entropy(target_model_logits, torch.tensor([j], device=device))
        model_diag_weight_grad_clean = (
            torch.autograd.grad(
                loss,
                simple_model.weight,
                create_graph=True,
            )[0]
            .detach()
            .clone()
            .squeeze()
            .cpu()
            .numpy()
        )
        grad_on_different_labels.append(model_diag_weight_grad_clean)

    return numpy.stack(grad_on_different_labels)
