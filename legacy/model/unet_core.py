import torch
import torch.nn as nn
import einops
from timm.models.convnext import ConvNeXtBlock

from model.core import AttentionGeneral, TBlockGeneral


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, subpixel=True):
        super().__init__()
        self.subpixel = subpixel
        if not subpixel:
            self.conv = torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        else:
            print("Using subpixel upsample")
            self.conv = torch.nn.Conv2d(
                in_channels, 4 * out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        if not self.subpixel:
            x = torch.nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
            x = self.conv(x)
        else:
            x = self.conv(x)
            x = nn.functional.pixel_shuffle(x, 2)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0, temb_channels=0, norm='default'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        ng = {'default': 32, 'instance': in_channels, 'group': in_channels // 16}[norm]
        self.norm1 = nn.GroupNorm(num_groups=ng, num_channels=in_channels)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
            self.temb_act = nn.GELU()

        ng = {'default': 32, 'instance': out_channels, 'group': out_channels // 16}[norm]
        self.norm2 = nn.GroupNorm(num_groups=ng, num_channels=out_channels)
        self.act2 = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.temb_act(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class MyConvNeXtBlockOne(ConvNeXtBlock):
    def __init__(self, in_channels, out_channels=None, dropout=0, temb_channels=0, norm='default'):
        assert dropout == 0
        assert temb_channels == 0
        assert norm == 'default'
        out_channels = in_channels if out_channels is None else out_channels

        super().__init__(out_channels, out_channels)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.nin_shortcut = None

    def forward(self, x, temb=None):
        assert temb is None
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return super().forward(x)


class MyConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0, temb_channels=0, norm='default'):
        super().__init__()
        self.block1 = MyConvNeXtBlockOne(in_channels, out_channels, dropout, temb_channels, norm)
        self.block2 = MyConvNeXtBlockOne(out_channels, out_channels, dropout, temb_channels, norm)
        self.block3 = MyConvNeXtBlockOne(out_channels, out_channels, dropout, temb_channels, norm)

    def forward(self, x, temb=None):
        x = self.block1(x, temb)
        x = self.block2(x, temb)
        x = self.block3(x, temb)
        return x


class ImageAttentionGeneral(AttentionGeneral):
    def forward(self, x, context=None):
        if self.context_dim is None:
            assert context is None
        else:
            assert context is not None

        # x : BCHW
        H, W = x.shape[-2:]
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        if context is not None and context.dim() == 4:
            context = einops.rearrange(context, 'b c h w -> b (h w) c')
        x = super().forward(x, context)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class ImageTBlockGeneral(TBlockGeneral):
    def __init__(self, *args, **kwargs):
        kwargs['mlp_ratio'] = 1
        super().__init__(*args, **kwargs)

    def forward(self, x, context=None):
        if self.context_dim is None:
            assert context is None
        else:
            assert context is not None
        H, W = x.shape[-2:]
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        if context is not None and context.dim() == 4:
            context = einops.rearrange(context, 'b c h w -> b (h w) c')
        x = super().forward(x, context)
        x = einops.rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class ImageAttentionGeneralReceptive(AttentionGeneral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rf = kwargs.pop('receptive_field', 7)
        assert self.rf % 2 == 1

    def forward(self, x, context=None, center=None):
        # x : BCHW
        # context: BCH'W'
        # center: B2HW in [-1, 1]

        if self.context_dim is None:
            assert context is None and center is None
        else:
            assert context is not None and center is not None

        B, _, xH, xW = x.shape
        M = self.num_heads
        D = self.head_dim

        if self.context_dim is None:
            qkv = self.qkv(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # B(3C)HW
            C = qkv.shape[1] // 3
            q = qkv[:, :C].permute(0, 2, 3, 1)  # BHWC
            kv = qkv[:, C:]  # B(2C)HW
            # B2HW
            center = torch.stack(
                torch.meshgrid(torch.linspace(-1, 1, xH), torch.linspace(-1, 1, xW))
            )[None].expand(B, -1, -1, -1)
        else:
            _, _, cH, cW = context.shape
            assert cH == cW

            kv = self.kv(context.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # B(2C)H'W'
            q = self.q(x.permute(0, 2, 3, 1))  # BHWC

        # --------------
        # sampling
        # --------------
        center = torch.nn.functional.interpolate(
            center, (xH, xW), mode='bilinear', align_corners=False
        )
        center = (center + 1) * ((cH - 1) / 2)  # uv to pixel index
        center = center.flatten(2, 3).permute(0, 2, 1)  # B(HW)2

        temp = torch.arange(-self.rf // 2 + 1, self.rf // 2 + 1, device=x.device).float()
        grid = torch.stack(torch.meshgrid(temp, temp, indexing='xy'), -1).flatten(0, 1)  # X2
        samples = center[:, :, None, :] + grid[None, None, :, :]  # B(HW)X2
        samples = samples * (2 / (cH - 1)) - 1  # pixel index to uv again

        # B(2C)(HW)X
        kv = torch.nn.functional.grid_sample(kv, samples, mode='bilinear', align_corners=False)
        # --------------

        k, v = kv.permute(0, 2, 3, 1).split(q.shape[-1], -1)  # B(HW)XC
        k = k.unflatten(-1, (M, D)).transpose(2, 3)  # B(HW)MXD
        v = v.unflatten(-1, (M, D)).transpose(2, 3)  # B(HW)MXD
        q = q.flatten(1, 2).unflatten(-1, (M, D))[:, :, :, None, :]  # B(HW)M1D

        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B(HW)M1X
        attn = attn.softmax(dim=-1)  # B(HW)M1X
        attn = self.attn_drop(attn)
        x = attn @ v  # B(HW)M1X @ B(HW)MXD gives B(HW)M1D

        x = x.squeeze(-2).flatten(2, 3)  # B(HW)C
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.permute(0, 2, 1).unflatten(-1, (xH, xW))
        return x


class ImageTBlockGeneralReceptive(TBlockGeneral):
    def __init__(self, *args, **kwargs):
        kwargs['mlp_ratio'] = 1
        kwargs['attn_cls'] = ImageAttentionGeneralReceptive
        super().__init__(*args, **kwargs)

    def forward(self, x, context=None, center=None):
        if self.context_dim is None:
            assert context is None and center is None
        else:
            assert context is not None and center is not None

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm1(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        x = x + self.attn(x, context, center)

        if self.context_dim is not None:
            x = einops.rearrange(x, 'b c h w -> b h w c')
            x = self.norm2(x)
            x = einops.rearrange(x, 'b h w c -> b c h w')
            x = x + self.cross_attn(x, context)

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm3(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        x = x + self.mlp(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        first_dim,
        num_levels,
        attn_start_level,
        num_res_blocks=2,
        dropout=0,
        use_attn_cls=ImageAttentionGeneral,
        norm='default',
    ):
        super().__init__()
        self.num_levels = num_levels
        self.num_res_blocks = num_res_blocks
        chans = [first_dim * (2**i) for i in range(num_levels)]

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, first_dim, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        NR = self.num_res_blocks
        conv_block = ResnetBlock
        for i in range(self.num_levels):
            layer = nn.Module()

            layer.downsample = Downsample(chans[i - 1], chans[i]) if i > 0 else None

            layer.blocks = nn.ModuleList(
                [conv_block(chans[i], chans[i], dropout=dropout, norm=norm) for _ in range(NR)]
            )
            if i >= attn_start_level:
                attns = [
                    use_attn_cls(chans[i], num_heads=chans[i] // first_dim, proj_drop=dropout)
                    for _ in range(NR)
                ]
                layer.attns = nn.ModuleList(attns)
            else:
                layer.attns = None

            self.down.append(layer)

    def forward(self, x, view_emb=None):
        # downsampling
        h = self.conv_in(x)
        skips = {}
        # skips = {f"{-1}_{0}": h}
        for i, layer in enumerate(self.down):
            if layer.downsample is not None:
                h = layer.downsample.forward(h)
            if view_emb is not None:
                h = h + view_emb[i]
            for j in range(self.num_res_blocks):
                h = layer.blocks[j].forward(h)
                if layer.attns is not None:
                    h = layer.attns[j].forward(h)
                skips[f"{i}_{j}"] = h
        # skips[f"{self.num_levels}_{0}"] = h
        return h, skips


class UNetDecoder(nn.Module):
    def __init__(
        self,
        first_dim,
        num_levels,
        attn_start_level,
        context_dim,
        num_res_blocks=2,
        dropout=0,
        use_attn_cls=ImageAttentionGeneral,
        num_skips=1,
        norm='default',
    ):
        """
        If cross_attend=False then it is a standard unet decoder. Attention with be self-attention.
        If cross_attend=True,
            if context_dim=-1, it will have multiscale context tokens from Unet context encoder.
            else cross attention with fixed context tokens (from context encoder bottleneck) at all layers.
        """
        super().__init__()
        self.num_levels = num_levels
        self.num_res_blocks = num_res_blocks
        self.context_dim = context_dim
        chans = [first_dim * (2**i) for i in range(num_levels)]

        self.up = nn.ModuleList()
        NR = self.num_res_blocks
        conv_block = ResnetBlock
        for i in list(range(self.num_levels))[::-1]:
            layer = nn.Module()
            layer.blocks = nn.ModuleList(
                [
                    conv_block(chans[i] * (1 + num_skips), chans[i], dropout=dropout, norm=norm)
                    for _ in range(NR)
                ]
            )
            if i >= attn_start_level:
                attns = [
                    use_attn_cls(chans[i], num_heads=chans[i] // first_dim, proj_drop=dropout)
                    for _ in range(NR)
                ]
                layer.attns = nn.ModuleList(attns)
            else:
                layer.attns = None
            layer.upsample = Upsample(chans[i], chans[i - 1]) if i > 0 else None

            self.up.append(layer)

    def forward(self, x, skips, context, uv, view_emb=None):
        if self.context_dim is None:
            assert context is None
        else:
            assert context is not None

        skips_dec = {}
        h = x
        for i, layer in enumerate(self.up):
            if view_emb is not None:
                h = h + view_emb[self.num_levels - 1 - i]
            for j in range(self.num_res_blocks):
                enc_idx = f"{self.num_levels - 1 - i}_{self.num_res_blocks - 1 - j}"
                h = torch.cat([h, skips[enc_idx]], dim=-3)
                h = layer.blocks[j].forward(h)
                if layer.attns is not None:
                    con = context if self.context_dim != -1 else context[enc_idx]
                    if uv is not None:
                        h = layer.attns[j].forward(h, con, uv)
                    else:
                        h = layer.attns[j].forward(h, con)
                skips_dec[enc_idx] = h
            if layer.upsample is not None:
                h = layer.upsample(h)

        return h, skips_dec


class UNet(nn.Module):
    """Vanilla UNet with attention feature."""

    def __init__(
        self,
        inp_chan,
        out_chan,
        first_dim,
        num_levels,
        attn_start_level,
        tblock,
        dropout=0,
        norm='default',
    ) -> None:
        super().__init__()

        self.first_dim = first_dim
        self.num_levels = num_levels
        self.attn_start_level = attn_start_level

        use_attn_cls = {False: ImageAttentionGeneral, True: ImageTBlockGeneral}[tblock]
        conv_block = ResnetBlock

        # input encoder
        self.inp_encoder = UNetEncoder(
            inp_chan,
            first_dim=self.first_dim,
            num_levels=self.num_levels,
            attn_start_level=self.attn_start_level,
            use_attn_cls=use_attn_cls,
            dropout=dropout,
        )
        last_dim = self.first_dim * (2**self.num_levels)
        self.inp_mid = nn.Sequential(
            Downsample(last_dim // 2, last_dim),
            conv_block(last_dim, last_dim, dropout=dropout, norm=norm),
            use_attn_cls(last_dim, num_heads=last_dim // self.first_dim, proj_drop=dropout)
            if self.attn_start_level <= self.num_levels
            else nn.Identity(),
            conv_block(last_dim, last_dim, dropout=dropout, norm=norm),
            Upsample(last_dim, last_dim // 2),
        )

        # input decoder
        self.inp_decoder = UNetDecoder(
            first_dim=self.first_dim,
            num_levels=self.num_levels,
            attn_start_level=self.attn_start_level,
            context_dim=None,
            use_attn_cls=use_attn_cls,
            dropout=dropout,
        )
        self.out_layer = nn.Conv2d(self.first_dim, out_chan, kernel_size=1)

        print("number of parameters", sum(p.numel() for p in self.inp_encoder.parameters()))
        print("number of parameters", sum(p.numel() for p in self.inp_mid.parameters()))
        print("number of parameters", sum(p.numel() for p in self.inp_decoder.parameters()))

    def forward(self, x):
        # x: B x C x H x W
        emb, skips = self.inp_encoder(x)  # BDpp
        emb = self.inp_mid(emb)
        emb, _ = self.inp_decoder(emb, skips, None, None)  # BDHW
        out = self.out_layer(emb)  # BCHW
        return out
