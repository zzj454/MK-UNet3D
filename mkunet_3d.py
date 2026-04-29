import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

__all__ = ['MKUNet3D', 'MKUNet3D_T', 'MKUNet3D_S', 'MKUNet3D_M', 'MKUNet3D_L']


# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = (module.kernel_size[0] * module.kernel_size[1]
                       * module.kernel_size[2] * module.out_channels)
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, (2.0 / fan_out) ** 0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d)):
        if module.weight is not None:
            nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'relu6':
        return nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'gelu':
        return nn.GELU()
    else:
        raise NotImplementedError(f'activation layer [{act}] is not found')


def channel_shuffle(x, groups):
    batchsize, num_channels, depth, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, depth, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, depth, height, width)
    return x


# ══════════════════════════════════════════════════════════════════
#  BQM Bottleneck（双向 GRU 实现，修复版）
#
#  修复内容：
#    1. state_dim=0 时完全跳过 BQM（self.bqm_enabled = False）
#    2. GRU hidden_size 由 state_dim 控制，不再硬编码 d_model//2
#    3. 输入投影 proj_in 将 d_model 压缩到 state_dim*2，
#       双向 GRU 输出 state_dim*2，再由 proj_out 还原到 d_model
#       → 不同 state_dim 产生不同参数量，测试可验证
#    4. 添加了 bqm_enabled 属性，供外部查询
# ══════════════════════════════════════════════════════════════════
class BQMBottleneck(nn.Module):
    """
    Bidirectional Quasiseparable Mixing Bottleneck（GRU 实现）

    用双向 GRU 高效替代手写 SSM for 循环，
    捕捉 3D 体积全局上下文（causal + anti-causal）。

    参数:
        d_model   : bottleneck 通道数（= channels[4]，默认160）
        state_dim : GRU 单向隐藏维度，控制长程建模能力
                    0  → 禁用 BQM（直通残差，等价于无 BQM）
                    16 → 极轻量，参数 +约0.05M
                    32 → 推荐，参数 +约0.10M  ← 默认
                    64 → 标准，参数 +约0.18M

    参数量计算（d_model=160, state_dim=32）：
        proj_in  : 160×64         = 10,240
        GRU      : 3×(64×32+32²)×2 = ~24,576（双向）
        proj_out : 64×160         = 10,240
        out_conv : 160×160 + IN   = ~25,760
        合计约 0.07M
    """

    def __init__(self, d_model: int, state_dim: int = 32):
        super().__init__()
        self.d_model   = d_model
        self.state_dim = state_dim
        self.bqm_enabled = (state_dim > 0)

        if not self.bqm_enabled:
            # state_dim=0：不创建任何子模块，forward 直接返回输入
            return

        # 输入投影：d_model → state_dim*2（压缩后交给 GRU，降低序列维度）
        self.norm     = nn.LayerNorm(d_model)
        self.proj_in  = nn.Linear(d_model, state_dim * 2, bias=False)

        # 双向 GRU：hidden_size=state_dim，双向输出 state_dim*2
        # cuDNN 并行实现，比 for 循环快 10~50 倍，梯度稳定
        self.gru = nn.GRU(
            input_size  = state_dim * 2,
            hidden_size = state_dim,
            batch_first = True,
            bidirectional = True,   # causal + anti-causal
        )

        # 输出投影：GRU 输出 state_dim*2 → d_model
        self.proj_out = nn.Linear(state_dim * 2, d_model, bias=False)

        # 1×1×1 卷积融合（恢复 3D 结构后再细化）
        self.out_conv = nn.Sequential(
            nn.Conv3d(d_model, d_model, kernel_size=1, bias=False),
            nn.InstanceNorm3d(d_model, affine=True),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        if not self.bqm_enabled:
            return
        nn.init.xavier_uniform_(self.proj_in.weight)
        nn.init.xavier_uniform_(self.proj_out.weight)
        # GRU 权重使用 PyTorch 默认初始化（orthogonal + uniform），无需额外设置

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, D, H, W]，C == d_model
        返回: [B, C, D, H, W]，与输入形状相同
        """
        if not self.bqm_enabled:
            return x   # 直通，等价于无 BQM

        B, C, D, H, W = x.shape
        residual = x

        # ── 1. 展平为序列 [B, L, C]，L = D*H*W ──
        seq = x.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, C)

        # ── 2. LayerNorm + 输入投影（降维到 state_dim*2）──
        seq = self.proj_in(self.norm(seq))   # [B, L, state_dim*2]

        # ── 3. 双向 GRU（并行，cuDNN 加速）──
        out, _ = self.gru(seq)               # [B, L, state_dim*2]

        # ── 4. 输出投影（还原到 d_model）──
        out = self.proj_out(out)             # [B, L, d_model]

        # ── 5. 重塑回 3D ──
        out = out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]

        # ── 6. 1×1×1 卷积细化 ──
        out = self.out_conv(out)

        # ── 7. 残差连接 ──
        return residual + out


# ══════════════════════════════════════════════════════════════════
#  原有模块（与原版完全相同）
# ══════════════════════════════════════════════════════════════════
class ChannelAttention3D(nn.Module):
    """3D 通道注意力模块"""
    def __init__(self, in_planes, ratio=16, activation='relu'):
        super().__init__()
        self.in_planes = in_planes
        if self.in_planes < ratio:
            ratio = self.in_planes
        self.reduced_channels = max(self.in_planes // ratio, 1)

        self.avg_pool   = nn.AdaptiveAvgPool3d(1)
        self.max_pool   = nn.AdaptiveMaxPool3d(1)
        self.fc1        = nn.Conv3d(in_planes, self.reduced_channels, 1, bias=False)
        self.activation = act_layer(activation, inplace=True)
        self.fc2        = nn.Conv3d(self.reduced_channels, in_planes, 1, bias=False)
        self.sigmoid    = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        for m in self.modules():
            _init_weights(m, '', scheme)

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention3D(nn.Module):
    """3D 空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 5, 7)
        self.conv    = nn.Conv3d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        for m in self.modules():
            _init_weights(m, '', scheme)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class GroupedAttentionGate3D(nn.Module):
    """3D 分组注意力门（InstanceNorm版）"""
    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super().__init__()
        if kernel_size == 1:
            groups = 1

        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.InstanceNorm3d(F_int, affine=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.InstanceNorm3d(F_int, affine=True)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1, affine=True),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        for m in self.modules():
            _init_weights(m, '', scheme)

    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MultiKernelDepthwiseConv3D(nn.Module):
    """3D 多核深度卷积（InstanceNorm版）"""
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super().__init__()
        self.in_channels = in_channels
        self.dw_parallel  = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, ks, stride,
                          ks // 2, groups=in_channels, bias=False),
                nn.InstanceNorm3d(in_channels, affine=True),
                act_layer(activation, inplace=True)
            )
            for ks in kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        for m in self.modules():
            _init_weights(m, '', scheme)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MultiKernelInvertedResidualBlock3D(nn.Module):
    """3D 多核倒置残差块（InstanceNorm版）"""
    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True,
                 add=True, kernel_sizes=[1, 3, 5], activation='relu6'):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c   = in_c
        self.out_c  = out_c
        self.add    = add
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = (stride == 1)

        self.ex_c = int(in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv3d(in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.InstanceNorm3d(self.ex_c, affine=True),
            act_layer(activation, inplace=True)
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv3D(
            self.ex_c, kernel_sizes, stride, activation, dw_parallel=dw_parallel
        )
        self.combined_channels = self.ex_c if add else self.ex_c * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv3d(self.combined_channels, out_c, 1, 1, 0, bias=False),
            nn.InstanceNorm3d(out_c, affine=True),
        )
        if self.use_skip_connection and (in_c != out_c):
            self.conv1x1 = nn.Conv3d(in_c, out_c, 1, 1, 0, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        for m in self.modules():
            _init_weights(m, '', scheme)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwout = self.multi_scale_dwconv(pout1)
        dout  = sum(dwout) if self.add else torch.cat(dwout, dim=1)
        dout  = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out   = self.pconv2(dout)
        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        return out


def mk_irb_bottleneck_3d(in_c, out_c, n, s, expansion_factor=2, dw_parallel=True,
                         add=True, kernel_sizes=[1, 3, 5], activation='relu6'):
    convs = [MultiKernelInvertedResidualBlock3D(
        in_c, out_c, s, expansion_factor, dw_parallel, add, kernel_sizes, activation
    )]
    for _ in range(1, n):
        convs.append(MultiKernelInvertedResidualBlock3D(
            out_c, out_c, 1, expansion_factor, dw_parallel, add, kernel_sizes, activation
        ))
    return nn.Sequential(*convs)


# ══════════════════════════════════════════════════════════════════
#  核心网络
# ══════════════════════════════════════════════════════════════════
class MK_UNet_3D(nn.Module):
    """
    3D MK-UNet + BQM Bottleneck

    在 encoder5 输出后、decoder1 之前插入 BQMBottleneck，
    用双向 GRU 为每个体素整合全局 3D 上下文。

    参数:
        bqm_state_dim : GRU 单向隐藏维度
                        0  → 禁用 BQM（纯卷积，与原版等价）
                        16 → 极轻量
                        32 → 推荐（默认）
                        64 → 标准
    """

    def __init__(self,
                 num_classes=1,
                 in_channels=2,
                 channels=[16, 32, 64, 96, 160],
                 depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 gag_kernel=3,
                 deep_supervision=False,
                 bqm_state_dim=32,
                 **kwargs):
        super().__init__()
        self.deep_supervision = deep_supervision

        # ── 编码器 ────────────────────────────────────────────
        self.encoder1 = mk_irb_bottleneck_3d(in_channels,  channels[0], depths[0], 1,
                                             expansion_factor, True, True, kernel_sizes)
        self.encoder2 = mk_irb_bottleneck_3d(channels[0],  channels[1], depths[1], 1,
                                             expansion_factor, True, True, kernel_sizes)
        self.encoder3 = mk_irb_bottleneck_3d(channels[1],  channels[2], depths[2], 1,
                                             expansion_factor, True, True, kernel_sizes)
        self.encoder4 = mk_irb_bottleneck_3d(channels[2],  channels[3], depths[3], 1,
                                             expansion_factor, True, True, kernel_sizes)
        self.encoder5 = mk_irb_bottleneck_3d(channels[3],  channels[4], depths[4], 1,
                                             expansion_factor, True, True, kernel_sizes)

        # ── BQM Bottleneck ────────────────────────────────────
        # bqm_state_dim=0 时内部直接 return x，不增加任何参数
        self.bqm = BQMBottleneck(d_model=channels[4], state_dim=bqm_state_dim)

        # ── 注意力门 ──────────────────────────────────────────
        self.AG1 = GroupedAttentionGate3D(channels[3], channels[3], channels[3] // 2,
                                          gag_kernel, channels[3] // 2)
        self.AG2 = GroupedAttentionGate3D(channels[2], channels[2], channels[2] // 2,
                                          gag_kernel, channels[2] // 2)
        self.AG3 = GroupedAttentionGate3D(channels[1], channels[1], channels[1] // 2,
                                          gag_kernel, channels[1] // 2)
        self.AG4 = GroupedAttentionGate3D(channels[0], channels[0], channels[0] // 2,
                                          gag_kernel, channels[0] // 2)

        # ── 解码器 ────────────────────────────────────────────
        self.decoder1 = mk_irb_bottleneck_3d(channels[4], channels[3], 1, 1,
                                             expansion_factor, True, True, kernel_sizes)
        self.decoder2 = mk_irb_bottleneck_3d(channels[3], channels[2], 1, 1,
                                             expansion_factor, True, True, kernel_sizes)
        self.decoder3 = mk_irb_bottleneck_3d(channels[2], channels[1], 1, 1,
                                             expansion_factor, True, True, kernel_sizes)
        self.decoder4 = mk_irb_bottleneck_3d(channels[1], channels[0], 1, 1,
                                             expansion_factor, True, True, kernel_sizes)
        self.decoder5 = mk_irb_bottleneck_3d(channels[0], channels[0], 1, 1,
                                             expansion_factor, True, True, kernel_sizes)

        # ── 通道/空间注意力 ───────────────────────────────────
        self.CA1 = ChannelAttention3D(channels[4], ratio=16)
        self.CA2 = ChannelAttention3D(channels[3], ratio=16)
        self.CA3 = ChannelAttention3D(channels[2], ratio=16)
        self.CA4 = ChannelAttention3D(channels[1], ratio=8)
        self.CA5 = ChannelAttention3D(channels[0], ratio=4)

        self.SA1 = SpatialAttention3D(kernel_size=7)
        self.SA2 = SpatialAttention3D(kernel_size=7)
        self.SA3 = SpatialAttention3D(kernel_size=7)
        self.SA4 = SpatialAttention3D(kernel_size=7)
        self.SA5 = SpatialAttention3D(kernel_size=7)

        # ── 输出头 ────────────────────────────────────────────
        self.out1 = nn.Conv3d(channels[2], num_classes, kernel_size=1)
        self.out2 = nn.Conv3d(channels[1], num_classes, kernel_size=1)
        self.out3 = nn.Conv3d(channels[0], num_classes, kernel_size=1)
        self.out4 = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        # ── Encoder ──────────────────────────────────────────
        out = F.max_pool3d(self.encoder1(x),   2, 2); t1 = out
        out = F.max_pool3d(self.encoder2(out), 2, 2); t2 = out
        out = F.max_pool3d(self.encoder3(out), 2, 2); t3 = out
        out = F.max_pool3d(self.encoder4(out), 2, 2); t4 = out
        out = F.max_pool3d(self.encoder5(out), 2, 2)

        # ── BQM 全局上下文建模 ─────────────────────────────────
        # patch=160³ 时: [B, 160, 5, 5, 5]，L=125，GRU 极高效
        # bqm_state_dim=0 时: 直通，无开销
        out = self.bqm(out)

        # ── Decoder Stage 1 ───────────────────────────────────
        out = self.CA1(out) * out
        out = self.SA1(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2,
                                   mode='trilinear', align_corners=False))
        t4  = self.AG1(g=out, x=t4)
        out = out + t4

        # ── Decoder Stage 2 ───────────────────────────────────
        out = self.CA2(out) * out
        out = self.SA2(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2,
                                   mode='trilinear', align_corners=False))
        p1  = F.interpolate(self.out1(out), scale_factor=8,
                            mode='trilinear', align_corners=False)
        t3  = self.AG2(g=out, x=t3)
        out = out + t3

        # ── Decoder Stage 3 ───────────────────────────────────
        out = self.CA3(out) * out
        out = self.SA3(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2,
                                   mode='trilinear', align_corners=False))
        p2  = F.interpolate(self.out2(out), scale_factor=4,
                            mode='trilinear', align_corners=False)
        t2  = self.AG3(g=out, x=t2)
        out = out + t2

        # ── Decoder Stage 4 ───────────────────────────────────
        out = self.CA4(out) * out
        out = self.SA4(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2,
                                   mode='trilinear', align_corners=False))
        p3  = F.interpolate(self.out3(out), scale_factor=2,
                            mode='trilinear', align_corners=False)
        t1  = self.AG4(g=out, x=t1)
        out = out + t1

        # ── Decoder Stage 5 ───────────────────────────────────
        out = self.CA5(out) * out
        out = self.SA5(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2,
                                   mode='trilinear', align_corners=False))
        p4  = self.out4(out)

        if self.deep_supervision:
            return [p4, p3, p2, p1]
        else:
            return [p4]


# ══════════════════════════════════════════════════════════════════
#  预定义配置
# ══════════════════════════════════════════════════════════════════
def MKUNet3D_T(num_classes=1, in_channels=2, **kwargs):
    return MK_UNet_3D(num_classes=num_classes, in_channels=in_channels,
                      channels=[4, 8, 16, 24, 32], **kwargs)


def MKUNet3D_S(num_classes=1, in_channels=2, **kwargs):
    return MK_UNet_3D(num_classes=num_classes, in_channels=in_channels,
                      channels=[8, 16, 32, 48, 80], **kwargs)


def MKUNet3D(num_classes=1, in_channels=2, **kwargs):
    return MK_UNet_3D(num_classes=num_classes, in_channels=in_channels,
                      channels=[16, 32, 64, 96, 160], **kwargs)


def MKUNet3D_M(num_classes=1, in_channels=2, **kwargs):
    return MK_UNet_3D(num_classes=num_classes, in_channels=in_channels,
                      channels=[32, 64, 128, 192, 320], **kwargs)


def MKUNet3D_L(num_classes=1, in_channels=2, **kwargs):
    return MK_UNet_3D(num_classes=num_classes, in_channels=in_channels,
                      channels=[64, 128, 256, 384, 512], **kwargs)


# ══════════════════════════════════════════════════════════════════
#  测试入口
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"测试设备: {device}\n")

    configs = [
        ('MKUNet3D（无BQM, s=0）',  dict(num_classes=3, in_channels=2,
                                          deep_supervision=True, bqm_state_dim=0)),
        ('MKUNet3D（+BQM s=16）',   dict(num_classes=3, in_channels=2,
                                          deep_supervision=True, bqm_state_dim=16)),
        ('MKUNet3D（+BQM s=32）',   dict(num_classes=3, in_channels=2,
                                          deep_supervision=True, bqm_state_dim=32)),
        ('MKUNet3D（+BQM s=64）',   dict(num_classes=3, in_channels=2,
                                          deep_supervision=True, bqm_state_dim=64)),
    ]

    x = torch.randn(1, 2, 128, 128, 128).to(device)

    for name, kwargs in configs:
        try:
            model = MKUNet3D(**kwargs).to(device)
            model.eval()
            total = sum(p.numel() for p in model.parameters())

            # BQM 参数量单独统计
            bqm_params = sum(p.numel() for n, p in model.named_parameters()
                             if 'bqm' in n)

            # GPU 预热
            with torch.no_grad():
                _ = model(x)
                if device == 'cuda':
                    torch.cuda.synchronize()

            # 正式计时（3次取平均）
            times = []
            with torch.no_grad():
                for _ in range(3):
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.time()
                    out = model(x)
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    times.append((time.time() - t0) * 1000)

            print(f"{name}")
            print(f"  总参数量:      {total/1e6:.4f}M")
            print(f"  BQM参数量:     {bqm_params/1e6:.4f}M")
            print(f"  基础参数量:    {(total-bqm_params)/1e6:.4f}M")
            print(f"  主输出形状:    {out[0].shape}")
            print(f"  推理时间:      {sum(times)/len(times):.1f}ms（3次均值，已预热）")
            print()

        except Exception as e:
            import traceback
            print(f"{name} 测试失败: {e}")
            traceback.print_exc()
            print()