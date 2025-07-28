from operator import attrgetter

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter


def score_cross_entropy(logits, label):
    return F.cross_entropy(logits["logit"], label)


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.to(trans.device)
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2), p=2)
    )
    return loss


def smooth_loss(pred, gold):
    eps = 0.2

    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()

    return loss


def get_loss(task, loss_name, data_batch, out, dataset_name):
    if task == "cls":
        label = data_batch["label"].to(out["logit"].device)
        if loss_name == "cross_entropy":
            if "label_2" in data_batch.keys():
                label_2 = data_batch["label_2"].to(out["logit"].device)
                if isinstance(data_batch["lam"], torch.Tensor):
                    loss = 0
                    for i in range(data_batch["pc"].shape[0]):
                        loss_tmp = (
                                smooth_loss(
                                    out["logit"][i].unsqueeze(0),
                                    label[i].unsqueeze(0).long(),
                                )
                                * (1 - data_batch["lam"][i])
                                + smooth_loss(
                            out["logit"][i].unsqueeze(0),
                            label_2[i].unsqueeze(0).long(),
                        )
                                * data_batch["lam"][i]
                        )
                        loss += loss_tmp
                    loss = loss / data_batch["pc"].shape[0]
                else:
                    loss = (
                            smooth_loss(out["logit"], label) * (1 - data_batch["lam"])
                            + smooth_loss(out["logit"], label_2) * data_batch["lam"]
                    )
            else:
                loss = F.cross_entropy(out["logit"], label)

        elif loss_name == "smooth":
            if "label_2" in data_batch.keys():
                label_2 = data_batch["label_2"].to(out["logit"].device)
                if isinstance(data_batch["lam"], torch.Tensor):
                    loss = 0
                    for i in range(data_batch["pc"].shape[0]):
                        loss_tmp = (
                                smooth_loss(
                                    out["logit"][i].unsqueeze(0),
                                    label[i].unsqueeze(0).long(),
                                )
                                * (1 - data_batch["lam"][i])
                                + smooth_loss(
                            out["logit"][i].unsqueeze(0),
                            label_2[i].unsqueeze(0).long(),
                        )
                                * data_batch["lam"][i]
                        )
                        loss += loss_tmp
                    loss = loss / data_batch["pc"].shape[0]
                else:
                    loss = (
                            smooth_loss(out["logit"], label) * (1 - data_batch["lam"])
                            + smooth_loss(out["logit"], label_2) * data_batch["lam"]
                    )
            else:
                loss = smooth_loss(out["logit"], label)
        else:
            assert False
    elif task == "cls_trans":
        label = data_batch["label"].to(out["logit"].device)
        trans_feat = out["trans_feat"]
        logit = out["logit"]
        if loss_name == "cross_entropy":
            if "label_2" in data_batch.keys():
                label_2 = data_batch["label_2"].to(out["logit"].device)
                if isinstance(data_batch["lam"], torch.Tensor):
                    loss = 0
                    for i in range(data_batch["pc"].shape[0]):
                        loss_tmp = (
                                smooth_loss(
                                    out["logit"][i].unsqueeze(0),
                                    label[i].unsqueeze(0).long(),
                                )
                                * (1 - data_batch["lam"][i])
                                + smooth_loss(
                            out["logit"][i].unsqueeze(0),
                            label_2[i].unsqueeze(0).long(),
                        )
                                * data_batch["lam"][i]
                        )
                        loss += loss_tmp
                    loss = loss / data_batch["pc"].shape[0]
                else:
                    loss = (
                            smooth_loss(out["logit"], label) * (1 - data_batch["lam"])
                            + smooth_loss(out["logit"], label_2) * data_batch["lam"]
                    )
            else:
                loss = F.cross_entropy(out["logit"], label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        elif loss_name == "smooth":
            if "label_2" in data_batch.keys():
                label_2 = data_batch["label_2"].to(out["logit"].device)
                if isinstance(data_batch["lam"], torch.Tensor):
                    loss = 0
                    for i in range(data_batch["pc"].shape[0]):
                        loss_tmp = (
                                smooth_loss(
                                    out["logit"][i].unsqueeze(0),
                                    label[i].unsqueeze(0).long(),
                                )
                                * (1 - data_batch["lam"][i])
                                + smooth_loss(
                            out["logit"][i].unsqueeze(0),
                            label_2[i].unsqueeze(0).long(),
                        )
                                * data_batch["lam"][i]
                        )
                        loss += loss_tmp
                    loss = loss / data_batch["pc"].shape[0]
                else:
                    loss = (
                            smooth_loss(out["logit"], label) * (1 - data_batch["lam"])
                            + smooth_loss(out["logit"], label_2) * data_batch["lam"]
                    )
            else:
                loss = smooth_loss(out["logit"], label)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        else:
            assert False
    else:
        assert False

    return loss


def knn_points(k, xyz, query, nsample=512):
    B, N, C = xyz.shape
    _, S, _ = query.shape  # S=1

    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis, np.newaxis, :], B, axis=0)
    sqrdists = square_distance(query, xyz)
    tmp = np.sort(sqrdists, axis=2)
    knn_dist = np.zeros((B, 1))
    for i in range(B):
        knn_dist[i][0] = tmp[i][0][k]
        group_idx[i][sqrdists[i] > knn_dist[i][0]] = N
    # group_idx[sqrdists > radius ** 2] = N
    # print("group idx : \n",group_idx)
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:, :, 0]
    group_first = np.repeat(tmp_idx[:, np.newaxis, :], nsample, axis=2)
    # repeat the first value of the idx in each batch
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def query_ball_point_for_rsmix(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample], S=1
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis, np.newaxis, :], B, axis=0)

    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N

    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:, :, 0]
    group_first = np.repeat(tmp_idx[:, np.newaxis, :], nsample, axis=2)
    # repeat the first value of the idx in each batch
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * np.matmul(src, dst.transpose(0, 2, 1))
    dist += np.sum(src ** 2, -1).reshape(B, N, 1)
    dist += np.sum(dst ** 2, -1).reshape(B, 1, M)

    return dist


def pts_num_ctrl(pts_erase_idx, pts_add_idx):
    """
    input : pts - to erase
            pts - to add
    output :pts - to add (number controled)
    """
    if len(pts_erase_idx) >= len(pts_add_idx):
        num_diff = len(pts_erase_idx) - len(pts_add_idx)
        if num_diff == 0:
            pts_add_idx_ctrled = pts_add_idx
        else:
            pts_add_idx_ctrled = np.append(
                pts_add_idx,
                pts_add_idx[np.random.randint(0, len(pts_add_idx), size=num_diff)],
            )
    else:
        pts_add_idx_ctrled = np.sort(
            np.random.choice(pts_add_idx, size=len(pts_erase_idx), replace=False)
        )
    return pts_add_idx_ctrled


def pgd(data_batch, model, task, loss_name, dataset_name, step=7, eps=0.05, alpha=0.01):
    model.eval()
    data = data_batch["pc"]
    adv_data = data.clone()
    adv_data = adv_data + (torch.rand_like(adv_data) * eps * 2 - eps)
    adv_data.detach()
    adv_data_batch = {}

    for _ in range(step):
        adv_data.requires_grad = True
        out = model(**{"pc": adv_data})
        adv_data_batch["pc"] = adv_data
        adv_data_batch["label"] = data_batch["label"]
        model.zero_grad()
        loss = get_loss(task, loss_name, adv_data_batch, out, dataset_name)
        loss.backward()
        with torch.no_grad():
            adv_data = adv_data + alpha * adv_data.grad.sign()
            delta = adv_data - data
            delta = torch.clamp(delta, -eps, eps)
            adv_data = (data + delta).detach_()

        return adv_data_batch
    else:
        return data_batch


def score_cross_entropy(logits, label):
    return F.cross_entropy(logits["logit"], label)


def smooth_risk_calculation(
        model, data, label, eps, step=1, score_function=None, track_modules=[], **kwargs
):
    device = data.device
    data, batch_index = replicate_tensor(data, step=step)
    label = replicate_label(label, step=step).to(label)
    if step > 1:
        data = data + torch.rand_like(data) * eps * 2 - eps
    if data.grad_fn is None:
        data = data.to(device)
        batch_index = batch_index.to(device)
        data.detach()
        data.requires_grad = True

    if not (data.grad is None):
        data.grad.zero_()
    track_modules = [attrgetter(module)(model) for module in track_modules]
    grads_in_hook = []
    model.zero_grad()

    def hook(module, input, output):
        grads_in_hook.append(input[0].detach())

    handles = [module.register_full_backward_hook(hook) for module in track_modules]

    out = model(**{"pc": data, "logits": False})
    loss = score_function(out, label=label)
    loss.backward()
    risk = torch.norm(data.grad, dim=-1)
    risk /= torch.norm(risk, dim=-1, keepdim=True)
    with torch.no_grad():
        for g in grads_in_hook[::-1]:
            r = torch.norm(g, dim=1)
            r /= torch.norm(r, dim=-1, keepdim=True)
            r[torch.isnan(r)] = 0
            risk += r
    for handle in handles:
        handle.remove()
    risk = sum_same_batch(risk, batch_index=batch_index) / step

    return risk


def replicate_tensor(input_tensor, step):
    new_tensor = input_tensor.repeat(step, 1, 1)
    batch_index = torch.arange(input_tensor.size(0)).repeat(step)
    return new_tensor, batch_index


def replicate_label(label, step):
    return label.repeat(step)


def sum_same_batch(input_tensor, batch_index):
    sum_tensor = torch_scatter.scatter_add(input_tensor, batch_index, dim=0)
    return sum_tensor
