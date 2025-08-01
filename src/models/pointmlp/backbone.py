import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points


def get_activation(activation):
    if activation.lower() == "gelu":
        return nn.GELU()
    elif activation.lower() == "rrelu":
        return nn.RReLU(inplace=True)
    elif activation.lower() == "selu":
        return nn.SELU(inplace=True)
    elif activation.lower() == "silu":
        return nn.SiLU(inplace=True)
    elif activation.lower() == "hardswish":
        return nn.Hardswish(inplace=True)
    elif activation.lower() == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


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
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = (
        torch.arange(B, dtype=torch.long)
        .to(device)
        .view(view_shape)
        .repeat(repeat_shape)
    )
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # device = xyz.device
    # B, N, C = xyz.shape
    # centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    # distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    # batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # for i in range(npoint):
    #     centroids[:, i] = farthest
    #     centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
    #     dist = torch.sum((xyz - centroid) ** 2, -1)
    #     distance = torch.min(distance, dist)
    #     farthest = torch.max(distance, -1)[1]
    # return centroids
    return sample_farthest_points(xyz, K=npoint)[1]


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = (
        torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    )
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(
        self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs
    ):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(
                f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor]."
            )
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(
                torch.ones([1, 1, 1, channel + add_channel])
            )
            self.affine_beta = nn.Parameter(
                torch.zeros([1, 1, 1, channel + add_channel])
            )

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        fps_idx = farthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat(
                [grouped_points, grouped_xyz], dim=-1
            )  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = (
                    torch.cat([new_points, new_xyz], dim=-1)
                    if self.use_xyz
                    else new_points
                )
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = (
                torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True)
                .unsqueeze(dim=-1)
                .unsqueeze(dim=-1)
            )
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat(
            [
                grouped_points,
                new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1),
            ],
            dim=-1,
        )
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=1, bias=True, activation="relu"
    ):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            self.act,
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(
        self,
        channel,
        kernel_size=1,
        groups=1,
        res_expansion=1.0,
        bias=True,
        activation="relu",
    ):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(
                in_channels=channel,
                out_channels=int(channel * res_expansion),
                kernel_size=kernel_size,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act,
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=int(channel * res_expansion),
                    out_channels=channel,
                    kernel_size=kernel_size,
                    groups=groups,
                    bias=bias,
                ),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=kernel_size,
                    bias=bias,
                ),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(
                    in_channels=int(channel * res_expansion),
                    out_channels=channel,
                    kernel_size=kernel_size,
                    bias=bias,
                ),
                nn.BatchNorm1d(channel),
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        blocks=1,
        groups=1,
        res_expansion=1,
        bias=True,
        activation="relu",
        use_xyz=True,
    ):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(
            in_channels, out_channels, bias=bias, activation=activation
        )
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(
                    out_channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(
        self,
        channels,
        blocks=1,
        groups=1,
        res_expansion=1,
        bias=True,
        activation="relu",
    ):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(
                    channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)
