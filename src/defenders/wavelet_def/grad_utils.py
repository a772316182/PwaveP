import torch

from .local_risk import extract_feature_grad
from .utils import smooth_risk_calculation, score_cross_entropy


def extract_grad_and_risk(
    data_batch,
    model,
    step=5,
    iter_num=1,
    eps=0.05,
    use_true=False,
    drop_rate=0.08,
):
    global label
    global track_modules
    global adv_data_batch

    point_num = data_batch["pc"].size()[1]
    all_drop_num = int(drop_rate) if drop_rate > 1 else int(drop_rate * point_num)
    drop_nums = [int(all_drop_num / iter_num) + 1 for _ in range(iter_num)]
    model.eval()
    _ = model.state_dict()
    ori_label = data_batch["label"].clone()

    if model.__class__.__name__ == "PointNet":
        track_modules = ["model.feat.fstn", "model.feat.conv3"]
    elif model.__class__.__name__ == "DGCNN":
        track_modules = ["model.conv5"]
    elif model.__class__.__name__ == "PCT":
        track_modules = ["model.conv_fuse"]
    elif model.__class__.__name__ == "PointNet2ClsMsg":
        track_modules = ["sa3.mlp_convs"]
    elif model.__class__.__name__ == "CurveNet":
        track_modules = ["cic11.conv1"]

    score_function = score_cross_entropy

    for i in range(iter_num):
        data = data_batch["pc"]
        drop_num = drop_nums[i]
        adv_data_ = data.clone()
        grad = 0.0
        if i == 0:
            out = model.forward(**{"pc": adv_data_, "topk": 1, "logits": True})
            label = (
                ori_label if use_true else out["logit"].max(dim=-1)[1].clone().detach()
            )
            adv_data_batch = {"label": label, "pc": adv_data_}

        grad = smooth_risk_calculation(
            model, adv_data_, label, eps, step, score_function, track_modules
        )
        local_grad = extract_feature_grad(adv_data_, k=20, mode="norm").to(grad)
        grad = grad + local_grad * 1.0
        adv_data_batch["pc"] = data

        device = adv_data_batch["pc"].device
        keep_ind = torch.topk(
            -grad, k=adv_data_batch["pc"].size()[1] - drop_num, dim=-1
        )[1].to(device)

        adv_data_batch["pc"].requires_grad = False

        with torch.no_grad():
            adv_data_batch["pc"] = adv_data_batch["pc"][
                torch.arange(keep_ind.size(0)).to(device), keep_ind.T.to(device), :
            ]
            adv_data_batch["pc"] = adv_data_batch["pc"].permute(1, 0, 2)

        data_batch = adv_data_batch
    adv_data_batch["label"] = ori_label
    adv_data_batch.update({"risk": grad.cpu().detach().numpy()})
    return adv_data_batch
