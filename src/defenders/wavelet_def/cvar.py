import torch

from .local_risk import calculate_local_risk
from .utils import smooth_risk_calculation, score_cross_entropy


def advanced_cvar(
        data_batch,
        model,
        step=5,
        iter_num=1,
        eps=0.05,
        use_true=False,
        drop_rate=0.08,
        drop_strategy="drop",
):
    # drop_strategy
    # (1) random: random compelete.
    # (2) zero: fill with 0
    # (3) drop: just directly drop.
    # (4) expand: multiply expand_rate
    # (5) random_expand: multiply random expand_rate
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
    elif model.__class__.__name__ == "Pct":
        track_modules = ["model.identity"]
    elif model.__class__.__name__ == 'PointNet2ClsMsg':
        track_modules = ["sa3.mlp_convs"]
    elif model.__class__.__name__ == 'CurveNet':
        track_modules = ['cic11.conv1']

    score_function = score_cross_entropy

    for i in range(iter_num):
        data = data_batch["pc"]
        drop_num = drop_nums[i]
        adv_data_ = data.clone()
        risk = 0.0
        if i == 0:
            out = model.forward(**{"pc": adv_data_, "topk": 1, "logits": True})
            label = (
                ori_label if use_true else out["logit"].max(dim=-1)[1].clone().detach()
            )
            adv_data_batch = {"label": label, "pc": adv_data_}

        risk = smooth_risk_calculation(model, adv_data_, label, eps, step, score_function, track_modules)
        l_risk = calculate_local_risk(adv_data_, k=20, mode="norm").to(risk)
        risk = risk + l_risk * 1.0
        adv_data_batch["pc"] = data

        if drop_strategy == "random":
            inds = torch.sort(risk, dim=1, descending=True)[1]
            drop_ind = inds[:, :drop_num]
            keep_ind = inds[:, drop_num:]
            rand_keep_ind = torch.randint(
                keep_ind.size(1), size=(keep_ind.size(0), drop_num)
            )
            rand_keep_ind = keep_ind[torch.arange(keep_ind.size(0)), rand_keep_ind.T]
            adv_data_batch["pc"].requires_grad = False
            with torch.no_grad():
                adv_data_batch["pc"][torch.arange(drop_ind.size(0)), drop_ind.T, :] = (
                    adv_data_batch["pc"][
                    torch.arange(drop_ind.size(0)), rand_keep_ind, :]
                )

        elif drop_strategy == "zero":
            drop_ind = torch.topk(risk, k=drop_num, dim=-1)[1]
            adv_data_batch["pc"].requires_grad = False
            with torch.no_grad():
                adv_data_batch["pc"][
                torch.arange(drop_ind.size(0)), drop_ind.T, :
                ] = 0.0

        elif drop_strategy == "drop":
            device = adv_data_batch["pc"].device
            keep_ind = torch.topk(
                -risk, k=adv_data_batch["pc"].size()[1] - drop_num, dim=-1
            )[1].to(device)

            adv_data_batch["pc"].requires_grad = False

            with torch.no_grad():
                adv_data_batch["pc"] = adv_data_batch["pc"][
                                       torch.arange(keep_ind.size(0)).to(device), keep_ind.T.to(device), :
                                       ]
                adv_data_batch["pc"] = adv_data_batch["pc"].permute(1, 0, 2)
        else:
            pass

        data_batch = adv_data_batch
    adv_data_batch["label"] = ori_label
    adv_data_batch.update({"risk": risk.cpu().detach().numpy()})
    return adv_data_batch
