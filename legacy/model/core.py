import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as weight_norm_fn


def linear_layer(inp_dim, out_dim, weight_norm):
    layer = nn.Linear(inp_dim, out_dim)
    if weight_norm:
        layer = weight_norm_fn(layer)
    return layer


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def my_groupnorm(num_hidden):
    num_g = 16
    assert num_hidden % num_g == 0
    return nn.GroupNorm(num_g, num_hidden)


def get_activation(activation):
    return {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
        'prelu': nn.PReLU,
        'selu': nn.SELU,
        'gelu': nn.GELU,
    }[activation]


def get_act_normalization(act_normalization):
    return {
        'none': nn.Identity,
        'batch': nn.BatchNorm1d,
        'layer': nn.LayerNorm,
        'group': my_groupnorm,
        'instance': nn.InstanceNorm1d,
    }[act_normalization]


class MyResBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        activation,
        act_normalization,
        time_dim,
        weight_norm=False,
    ):
        super().__init__()
        self.fc1 = linear_layer(hidden_dim, hidden_dim, weight_norm)
        self.bn1 = act_normalization(hidden_dim)
        self.act1 = activation()
        self.fc2 = linear_layer(hidden_dim, hidden_dim, weight_norm)
        self.bn2 = act_normalization(hidden_dim)
        self.act2 = activation()
        self.time_fc = None
        if time_dim is not None:
            self.time_fc = nn.Sequential(activation(), nn.Linear(time_dim, 2 * hidden_dim))

    def forward(self, x, t=None):
        out = self.bn1(self.fc1(x))

        if t is not None:
            t = self.time_fc(t)
            scale, shift = t.chunk(2, dim=-1)
            out = out * (scale + 1) + shift

        out = self.act1(out)
        out = self.bn2(self.fc2(out))
        out = self.act2(out + x)
        return out


class PaddedNorm(nn.Module):
    def __init__(self, d, act_normalization):
        super().__init__()
        self.d = d
        self.nd = ((d + 15) // 16) * 16
        self.pad = torch.nn.ConstantPad1d([0, self.nd - self.d], 0)
        self.norm_fn = act_normalization(self.nd)

    def forward(self, x):
        x = self.pad(x)
        x = self.norm_fn(x)
        return x[..., : self.d]


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MyResMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        activation,
        num_hiddens,
        act_normalization,
        time_dim,
        weight_norm=False,
    ):
        super().__init__()
        assert num_hiddens > 0

        layers = [
            linear_layer(input_dim, hidden_dim, weight_norm),
            act_normalization(hidden_dim),
            activation(),
        ]
        self.inp_block = nn.Sequential(*layers)

        layers = [
            MyResBlock(hidden_dim, activation, act_normalization, time_dim, weight_norm)
            for _ in range(num_hiddens - 1)
        ]
        self.block = mySequential(*layers)

        self.out_block = linear_layer(hidden_dim, output_dim, weight_norm)

    def forward(self, x, t):
        x = self.inp_block(x)
        x = self.block(x, t)
        x = self.out_block(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        num_heads = max(num_heads, 1)
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        num_heads = max(num_heads, 1)
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, CN, _ = context.shape
        kv = (
            self.kv(context).reshape(B, CN, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        )
        q = self.q(x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        (q,) = q.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DecBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        context_dim=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            dim if context_dim is None else context_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm3 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x, context):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class AttentionGeneral(nn.Module):
    def __init__(
        self,
        dim,
        context_dim=None,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        num_heads = max(num_heads, 1)
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.context_dim = context_dim

        if self.context_dim is None:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        B, N, C = x.shape
        if self.context_dim is None:
            assert context is None
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            assert context is not None
            _, CN, _ = context.shape
            kv = (
                self.kv(context)
                .reshape(B, CN, 2, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q = self.q(x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)
            (q,) = q.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TBlockGeneral(nn.Module):
    def __init__(
        self,
        dim,
        context_dim=None,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        proj_drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        mlp_layer=Mlp,
        attn_cls=AttentionGeneral,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_cls(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.context_dim = context_dim
        if context_dim is not None:
            self.norm2 = norm_layer(dim)
            self.cross_attn = attn_cls(
                dim,
                context_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )

        self.norm3 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x, context=None):
        x = x + self.attn(self.norm1(x))
        if self.context_dim is not None:
            assert context is not None
            x = x + self.cross_attn(self.norm2(x), context)
        else:
            assert context is None
        x = x + self.mlp(self.norm3(x))
        return x


class SASA_Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=7, num_heads=8):
        super(SASA_Layer, self).__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.dk = self.dv = in_channels
        self.dkh = self.dk // self.num_heads
        self.dvh = self.dv // self.num_heads

        assert self.dk % self.num_heads == 0, "dk should be divided by num_heads."
        assert self.dk % self.num_heads == 0, "dv should be divided by num_heads."
        assert self.kernel_size % 2 == 1, "kernel_size should be odd number."

        self.k_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1)
        self.q_conv = nn.Conv2d(self.dk, self.dk, kernel_size=1)
        self.v_conv = nn.Conv2d(self.dv, self.dv, kernel_size=1)

        # # Positional encodings
        self.pos_h = nn.Parameter(torch.randn(self.num_heads, self.kernel_size) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(self.num_heads, self.kernel_size) * 0.02)

    def forward(self, x, cx=None):
        batch_size, _, height, width = x.size()
        if cx is None:
            cx = x

        # Compute k, q, v
        padded_cx = nn.functional.pad(
            cx,
            [
                (self.kernel_size - 1) // 2,
                (self.kernel_size - 1) - ((self.kernel_size - 1) // 2),
                (self.kernel_size - 1) // 2,
                (self.kernel_size - 1) - ((self.kernel_size - 1) // 2),
            ],
        )
        k = self.k_conv(padded_cx)
        q = self.q_conv(x)
        v = self.v_conv(padded_cx)

        # Unfold patches into [BS, num_heads*depth, horizontal_patches, vertical_patches, kernel_size, kernel_size]
        k = k.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        v = v.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)

        # Reshape into [BS, num_heads, horizontal_patches, vertical_patches, depth_per_head, kernel_size*kernel_size]
        # BMdHWKK -> BMHWd(KK)
        k = k.unflatten(1, (self.num_heads, self.dkh)).permute(0, 1, 3, 4, 2, 5, 6).flatten(-2, -1)
        v = v.unflatten(1, (self.num_heads, self.dvh)).permute(0, 1, 3, 4, 2, 5, 6).flatten(-2, -1)
        # k = k.reshape(batch_size, self.num_heads, height, width, self.dkh, -1)
        # v = v.reshape(batch_size, self.num_heads, height, width, self.dvh, -1)

        # Reshape into [BS, num_heads, height, width, depth_per_head, 1]
        # BMdHW -> BMHWd1
        q = q.unflatten(1, (self.num_heads, self.dkh)).permute(0, 1, 3, 4, 2).unsqueeze(-1)
        # q = q.reshape(batch_size, self.num_heads, height, width, self.dkh, 1)

        qk = torch.matmul(q.transpose(4, 5), k)  # BMHW1(KK)
        qk = qk.reshape(
            batch_size, self.num_heads, height, width, self.kernel_size, self.kernel_size
        )  # BMHWKK

        # # Add positional encoding
        if self.pos_h is not None and self.pos_w is not None:
            # pylint: disable=unsubscriptable-object
            qk = qk + self.pos_h[None, :, None, None, :, None]
            qk = qk + self.pos_w[None, :, None, None, None, :]

        qk = qk.flatten(-2, -1).unsqueeze(-2)  # BMHW1(KK)
        weights = torch.softmax(qk, dim=-1)

        attn_out = torch.matmul(weights, v.transpose(4, 5)).squeeze(-2)  # BMHWd
        attn_out = attn_out.permute(0, 1, 4, 2, 3).flatten(1, 2)  # B(Md)HW
        return attn_out


class SASAGeneral(nn.Module):
    def __init__(self, in_channels, kernel_size=7, num_heads=8):
        super().__init__()
        dim = in_channels
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SASA_Layer(
            in_channels=in_channels, kernel_size=kernel_size, num_heads=num_heads
        )

        self.norm3 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(2 * dim, dim, kernel_size=1)

    def forward(self, x, context=None):
        # x: BDHW
        sc = x
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.attn(x, context)
        x = sc + x

        sc = x
        x = self.norm3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = sc + x

        return x


class TwoModelWrapper(nn.Module):
    def __init__(self, model_cls, n, *args, **kwargs):
        super().__init__()
        self.n = n
        self.mod1 = model_cls(*args, **kwargs)
        self.mod2 = model_cls(*args, **kwargs)

    def forward(self, xs, iter_idx):
        if iter_idx <= self.n:
            return self.mod1(xs, iter_idx)
        else:
            return self.mod2(xs, iter_idx)


class Overfit(torch.nn.Module):
    def __init__(self, num_pred_params, *args, **kwargs):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(1, num_pred_params) * 0.02)

    def forward(self, *args, **kwargs):
        return self.param


def flip_lr_every_other(x):
    assert x.dim() == 5  # BNCHW
    N = x.shape[1]
    x = torch.stack([x[:, i] if i % 2 == 0 else x[:, i].flip(-1) for i in range(N)], 1)
    return x


def flip_u_every_other(x):
    assert x.dim() == 5  # BN2HW
    assert x.shape[2] == 2
    N = x.shape[1]
    rx = []
    for i in range(N):
        if i % 2 == 0:
            rx.append(x[:, i])
        else:
            rx.append(torch.stack([-x[:, i, 0], x[:, i, 1]], 1))
    rx = torch.stack(rx, 1)  # BN2HW
    return rx


def gaussian_kernel_1d(sigma: float, num_sigmas: float = 2.0) -> torch.Tensor:
    radius = math.ceil(num_sigmas * sigma)
    support = torch.arange(-radius, radius + 1, dtype=torch.float)
    kernel = torch.distributions.Normal(loc=0, scale=sigma).log_prob(support).exp_()
    # Ensure kernel weights sum to 1, so that image brightness is not altered
    return kernel.mul_(1 / kernel.sum())


def gaussian_filter_2d(img: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_1d = gaussian_kernel_1d(sigma)  # Create 1D Gaussian kernel
    kernel_1d = kernel_1d.to(img.device).to(img.dtype)

    padding = len(kernel_1d) // 2  # Ensure that image size does not change
    # Convolve along columns and rows
    num_c = img.shape[1]
    img = img.flatten(0, 1)[:, None]
    img = nn.functional.conv2d(img, weight=kernel_1d.view(1, 1, -1, 1), padding=(padding, 0))
    img = nn.functional.conv2d(img, weight=kernel_1d.view(1, 1, 1, -1), padding=(0, padding))
    img = img[:, 0].unflatten(0, (-1, num_c))
    return img
