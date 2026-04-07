import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34"""

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BasicBlock.expansion,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers"""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels * BottleNeck.expansion,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * BottleNeck.expansion,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, c=3):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        output = self.conv1(x)
        output = self.conv2_x(output)
        # pooling in conv3
        output = self.conv3_x(output)
        # pooling in conv4
        output = self.conv4_x(output)
        # pooling in conv5
        output = self.conv5_x(output)
        # avg pooling to [1, 1]
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet18():
    """return a ResNet 18 object"""
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34():
    """return a ResNet 34 object"""
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50(num_cls=100, c=3):
    """return a ResNet 50 object"""
    return ResNet(BottleNeck, [3, 4, 6, 3], num_cls, c)


def resnet101():
    """return a ResNet 101 object"""
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152():
    """return a ResNet 152 object"""
    return ResNet(BottleNeck, [3, 8, 36, 3])


class LatticeGen(nn.Module):
    def __init__(self, s, normal_channel):
        super(LatticeGen, self).__init__()

        d = 3
        self.d = d
        # self.d1 = self.d + 1

        self.d1 = self.d
        self.size2d = s
        self.normal_channel = normal_channel

        self.elevate_mat = (
            torch.FloatTensor([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
            / torch.tensor(6.0).sqrt()
        )

        # canonical
        canonical = torch.arange(self.d1, dtype=torch.long)[None, :].repeat(self.d1, 1)
        # (d+1, d+1)
        for i in range(1, self.d1):
            canonical[-i:, i] = i - self.d1
        self.canonical = canonical

        self.dim_indices = torch.arange(self.d1, dtype=torch.long)[:, None]

    def get_keys_and_barycentric(self, pc, is_2nd_trans=False):
        """
        :param pc: (self.d, N -- undefined)
        :return:
        """
        batch_size = pc.size(0)
        num_points = pc.size(-1)
        device = pc.device

        point_indices = torch.arange(num_points, dtype=torch.long, device=device)[
            None, None, :
        ]
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)[
            :, None, None
        ]

        if is_2nd_trans:
            elevated = torch.matmul(self.elevate_mat2, pc)
        else:
            elevated = torch.matmul(
                self.elevate_mat.to(device), pc
            )  # * self.expected_std  # (d+1, N)

        greedy = torch.round(elevated / self.d1) * self.d1  # (d+1, N)

        el_minus_gr = elevated - greedy

        rank = torch.sort(el_minus_gr, dim=1, descending=True)[1]

        # the following advanced indexing is different in PyTorch 0.4.0 and 1.0.0
        # rank[rank, point_indices] = self.dim_indices  # works in PyTorch 0.4.0 but fail in PyTorch 1.x
        index = rank.clone()

        rank[batch_indices, index, point_indices] = self.dim_indices.to(
            device
        )  # works both in PyTorch 1.x(has tested in PyTorch 1.2) and PyTorch 0.4.0
        del index

        remainder_sum = greedy.sum(dim=1, keepdim=True) / self.d1

        rank_float = rank.type(torch.float32)
        cond_mask = (
            (rank_float >= self.d1 - remainder_sum) * (remainder_sum > 0)
            + (rank_float < -remainder_sum) * (remainder_sum < 0)
        ).type(torch.float32)
        sum_gt_zero_mask = (remainder_sum > 0).type(torch.float32)
        sum_lt_zero_mask = (remainder_sum < 0).type(torch.float32)
        sign_mask = -1 * sum_gt_zero_mask + sum_lt_zero_mask

        greedy += self.d1 * sign_mask * cond_mask
        rank += (self.d1 * sign_mask * cond_mask).type_as(rank)
        rank += remainder_sum.type(torch.long)

        # barycentric
        el_minus_gr = elevated - greedy
        greedy = greedy.type(torch.long)

        barycentric = torch.zeros(
            (batch_size, self.d1 + 1, num_points), dtype=torch.float32, device=device
        )

        barycentric[batch_indices, self.d - rank, point_indices] += el_minus_gr
        barycentric[batch_indices, self.d1 - rank, point_indices] -= el_minus_gr
        barycentric /= self.d1
        barycentric[batch_indices, 0, point_indices] += (
            1.0 + barycentric[batch_indices, self.d1, point_indices]
        )
        barycentric = barycentric[:, :-1, :]
        # canonical[rank, :]: [d1, num_pts, d1]
        #                     (d1 dim coordinates) then (d1 vertices of a simplex)
        keys = (
            greedy[:, :, :, None] + self.canonical.to(device)[rank, :]
        )  # (d1, num_points, d1)
        # rank: rearrange the coordinates of the canonical

        del (
            elevated,
            greedy,
            rank,
            remainder_sum,
            rank_float,
            cond_mask,
            sum_gt_zero_mask,
            sum_lt_zero_mask,
            sign_mask,
        )
        return keys, barycentric, el_minus_gr

    # def get_filter_size(self, radius):
    #     return (radius + 1) ** self.d1 - radius ** self.d1

    def convert2Dcoord(self, coord, batch_size, num_pts):
        offset = coord.min(dim=2)[0]

        coord -= offset.view(batch_size, -1, 1).expand(
            batch_size, -1, self.d1 * num_pts
        )
        pts_pick = (-offset) % 3
        return coord, pts_pick

    def get2D(self, coord, tmp, pts_pick, batch_size):
        """
        coord:   [B, 2, d1 * num_pts]
        tmp:     [B, d1 * num_pts, C]
        pts_pick:[B, 2]
        返回:    filter_2d: [B, C, H_out, W_out]  # 注意通道在前
        """
        device = coord.device
        B, _, L = coord.shape
        tmp = tmp.contiguous()
        _, L2, C = tmp.shape
        assert L == L2, f"coord last dim {L} != tmp second dim {L2}"

        H = self.size2d
        W = self.size2d
        d1 = self.d1
        H_out = H // d1
        W_out = W // d1

        # 1) 坐标 & 有效 mask
        coord_int = coord.long()
        x = coord_int[:, 0, :]  # [B, L]
        y = coord_int[:, 1, :]  # [B, L]

        in_range_x = (x >= 0) & (x < H)
        in_range_y = (y >= 0) & (y < W)
        in_range = in_range_x & in_range_y  # [B, L]

        # 有效点对应的特征
        v = tmp.reshape(B * L, C)[in_range.reshape(-1)].contiguous()  # [N_valid, C]

        # 有效点对应的 batch 下标
        b_idx_full = torch.arange(B, device=device)[:, None].expand(B, L)  # [B, L]
        b_idx = b_idx_full[in_range].contiguous()  # [N_valid]

        # 如果一个有效点都没有，直接返回零
        if v.numel() == 0:
            filter_2d = torch.zeros(
                (B, C, H_out, W_out),
                dtype=tmp.dtype,
                device=device,
            )
            if not self.normal_channel:
                filter_2d[filter_2d > 0] = 1.0
            return filter_2d

        x = x[in_range]  # [N_valid]
        y = y[in_range]

        # 2) 对齐到采样网格（stride = d1, 起点 = pts_pick）
        h0 = pts_pick[:, 0]  # [B]
        w0 = pts_pick[:, 1]  # [B]
        h0_pts = h0[b_idx]  # [N_valid]
        w0_pts = w0[b_idx]

        in_sample_range_h = (x >= h0_pts) & (x < h0_pts + H_out * d1)
        in_sample_range_w = (y >= w0_pts) & (y < w0_pts + W_out * d1)
        on_grid_h = ((x - h0_pts) % d1) == 0
        on_grid_w = ((y - w0_pts) % d1) == 0

        valid = in_sample_range_h & in_sample_range_w & on_grid_h & on_grid_w

        if not valid.any():
            filter_2d = torch.zeros(
                (B, C, H_out, W_out),
                dtype=tmp.dtype,
                device=device,
            )
            if not self.normal_channel:
                filter_2d[filter_2d > 0] = 1.0
            return filter_2d

        x = x[valid]
        y = y[valid]
        v = v[valid]  # [N_used, C]
        b_idx = b_idx[valid]
        h0_pts = h0_pts[valid]
        w0_pts = w0_pts[valid]

        # 3) 计算在下采样网格 (H_out, W_out) 上的索引
        h_idx = (x - h0_pts) // d1  # [N_used]
        w_idx = (y - w0_pts) // d1  # [N_used]

        # 4) 在 [B, C, H_out, W_out] 上累加
        filter_2d = torch.zeros(
            (B, C, H_out, W_out),
            dtype=tmp.dtype,
            device=device,
        )

        # 展平成 [B * H_out * W_out, C] 后做 index_add_
        # 这里 C 在前，我们先把通道挪到最后方便 index_add_
        filter_2d_flat = (
            filter_2d.permute(0, 2, 3, 1).contiguous().view(-1, C)
        )  # [B*H_out*W_out, C]

        flat_idx = b_idx * (H_out * W_out) + h_idx * W_out + w_idx  # [N_used]
        filter_2d_flat.index_add_(0, flat_idx, v)

        # 再改回 [B, C, H_out, W_out]
        filter_2d = (
            filter_2d_flat.view(B, H_out, W_out, C).permute(0, 3, 1, 2).contiguous()
        )

        if not self.normal_channel:
            filter_2d[filter_2d > 0] = 1.0

        return filter_2d

    def forward(self, pc1, features):
        # with torch.no_grad():
        # keys, bary, el_minus_gr = self.get_single(pc1[0])
        keys, in_barycentric, _ = self.get_keys_and_barycentric(pc1)
        # keys2, in_barycentric2, _2 = self.get_keys_and_barycentric(pc1, True)

        d = 2

        batch_size = features.size(0)
        num_pts = features.size(-1)
        # batch_indices = torch.arange(batch_size, dtype=torch.long)

        # batch_indices = batch_indices.pin_memory()

        # convert to 2d image
        # coord [3, d * num_pts]: [d] + [d] + ... + [d]
        coord = keys[:, :d].view(batch_size, d, -1)
        # import pdb; pdb.set_trace()

        coord, pts_pick = self.convert2Dcoord(coord, batch_size, num_pts)
        # tmp: [batch, d, d, num]
        # d: coordinates of points; then d: vertices of each simplex
        tmp = in_barycentric[:, None, :, :] * features[:, :, None, :]
        # tmp: [d, d * num]
        # d * num_pts: [d] + [d] + ... + [d]
        cc = tmp.size(1)
        tmp = (
            tmp.permute(0, 1, 3, 2)
            .contiguous()
            .view(batch_size, cc, -1)
            .permute(0, 2, 1)
        )
        filter_2d = self.get2D(coord, tmp, pts_pick, batch_size)
        return filter_2d, None, [filter_2d, keys.view(batch_size, 3, -1)]

    # return filter_2d, filter_2d_2, [filter_2d]#[splatted_2d, keys.view(batch_size, 3, -1)]
