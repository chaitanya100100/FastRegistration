import torch
from torch import nn
from randaugment import MyAugment
from .swin_core import SwinUNetEncoder, SwinUNetDecoder

# from .pos_encoding import PositionalEncodingPermute2D
from .core import flip_u_every_other


class STSwinUVMap(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        model_kwargs = kwargs['swin_uv_kwargs']
        img_size = (
            model_kwargs['render_size']
            if model_kwargs['render_size'] is not None
            else kwargs['render_size']
        )
        self.cameras = (
            model_kwargs['cameras'] if model_kwargs['cameras'] is not None else kwargs['cameras']
        )
        assert img_size[0] == img_size[1]
        first_dim = model_kwargs['first_dim']
        patch_size = model_kwargs['patch_size']
        window_size = model_kwargs['window_size']
        head_dim = model_kwargs['head_dim']
        depths = model_kwargs['depths']
        view_emb = model_kwargs['view_emb']
        self.patch_size = patch_size

        self.mod1 = SwinUNetEncoder(
            img_size[0], 1, patch_size, window_size, first_dim, head_dim, depths
        )
        self.mod2 = SwinUNetDecoder(
            img_size[0], 1, patch_size, window_size, first_dim, head_dim, depths
        )

        ps, fd = patch_size, first_dim
        self.out_layer = nn.Linear(fd, ps * ps * (2 + 1))
        self.num_layers = self.mod1.tsfm.num_layers

        self.view_pos_emb = None
        if view_emb:
            self.view_pos_emb = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(len(self.cameras), fd * 2**i) * 0.02)
                    for i in range(self.num_layers)
                ]
            )

    def forward(self, x, view_idx=None):
        B, N, _, H, W = x.shape
        x = x.flatten(0, 1)
        ps = self.patch_size

        view_emb = None
        if self.view_pos_emb is not None:
            if view_idx is None:
                view_idx = list(range(len(self.cameras)))
            if isinstance(view_idx[0], str):
                view_idx = [self.cameras.index(v) for v in view_idx]
            assert len(view_idx) == N
            view_emb = [
                self.view_pos_emb[i][view_idx][None].expand(B, -1, -1).flatten(0, 1)[:, None, :]
                for i in range(self.num_layers)
            ]

        x, skips = self.mod1(x, view_emb=view_emb)
        x, _ = self.mod2(x, skips, view_emb=view_emb)
        x = self.out_layer(x)  # (BN)(H/4W/4)(4*4*3)
        x = x.unflatten(1, (H // ps, W // ps)).permute(0, 3, 1, 2)  # (BN)(4*4*3)(H/4)(H/4)
        pred_uvm = nn.functional.pixel_shuffle(x, ps)  # (BN)3HW

        pred_uvm = pred_uvm.unflatten(0, (B, N))  # BN(2+1)HW
        pred_uv, pred_mask_logits = pred_uvm[:, :, :2], pred_uvm[:, :, 2:]  # BN2HW, BN1HW
        pred_uv = flip_u_every_other(pred_uv)
        return pred_uv, pred_mask_logits


class STSwinRGB(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.tex_inp_size = kwargs.get('tex_inp_size')
        self.cond_type = kwargs.get('cond_type')
        self.inp_img_in_cond = kwargs.get('inp_img_in_cond')
        self.num_cond_images = len(kwargs.get('cond_frame_num'))
        num_views = len(kwargs['cameras'])

        model_kwargs = kwargs['swin_rgb_kwargs']
        img_size = (
            model_kwargs['render_size']
            if model_kwargs['render_size'] is not None
            else kwargs['render_size']
        )
        assert img_size[0] == img_size[1]
        num_views = len(kwargs['cameras'])

        first_dim = model_kwargs['first_dim']
        patch_size = model_kwargs['patch_size']
        window_size = model_kwargs['window_size']
        head_dim = model_kwargs['head_dim']
        depths = model_kwargs['depths']
        view_emb = model_kwargs['view_emb']
        mlp_ratio = model_kwargs['mlp_ratio']
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.connection_type = model_kwargs['connection_type']
        assert self.connection_type in ["cross_attn", "concat"]

        M = self.num_cond_images
        self.tex_cnn = SwinUNetEncoder(
            img_size[0],
            3 * M + int(self.inp_img_in_cond),
            patch_size=patch_size,
            window_size=window_size,
            first_dim=first_dim,
            head_dim=head_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
        )

        self.st_cnn = SwinUNetEncoder(
            img_size[0],
            1,
            patch_size=patch_size,
            window_size=window_size,
            first_dim=first_dim,
            head_dim=head_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
        )

        embed_dim = [int(first_dim * 2**i) for i in range(len(depths))]
        # self.cond_pos_emb = nn.ModuleList([PositionalEncodingPermute2D(e) for e in embed_dim])

        self.rgb_decoder = SwinUNetDecoder(
            img_size[0],
            1,
            patch_size=patch_size,
            window_size=window_size,
            first_dim=first_dim,
            head_dim=head_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
            num_skips=1 + int(self.connection_type == "concat"),
            use_context=(self.connection_type == "cross_attn"),
        )
        self.rgb_layer = nn.Linear(first_dim, (self.patch_size**2) * 3)

        self.view_pos_emb = None
        if view_emb:
            self.view_pos_emb = nn.ParameterList(
                [nn.Parameter(torch.randn(num_views, e) * 0.02) for e in embed_dim]
            )

        print("number of parameters", sum(p.numel() for p in self.tex_cnn.parameters()))
        print("number of parameters", sum(p.numel() for p in self.st_cnn.parameters()))
        print("number of parameters", sum(p.numel() for p in self.rgb_decoder.parameters()))
        self.aug_module = None
        image_aug = kwargs.get("image_aug")
        if image_aug != "none":
            print("Using image augmentation", image_aug)
            aug_scale = int(image_aug.split("_")[1])
            self.aug_module = MyAugment(aug_scale)

    def forward(self, x, id_cond, uv_outputs):
        # x: B x N x C x H x W
        # id_cond: B x M x C x H x W
        # uv: B x N x 2 x H x W

        B, N, _, H, W = x.shape
        M = id_cond.shape[2]
        x = x.flatten(0, 1)
        uv = uv_outputs['pred_uv'].flatten(0, 1)  # (BN)2HW
        id_cond = id_cond.flatten(0, 1).flatten(1, 2)  # (BN)(M3)HW

        if '_uv_' in self.cond_type:
            id_cond = nn.functional.grid_sample(
                id_cond, uv.permute(0, 2, 3, 1), align_corners=False
            )  # (BN)(M3)HW with spatially aligned with image
            mask = (uv_outputs['pred_mask_logits'] > 0).float().flatten(0, 1)
            id_cond = mask * id_cond  # (BN)(M3)HW

        if self.aug_module is not None and self.training:
            id_cond = id_cond.unflatten(1, (M, 3)).flatten(0, 1)  # (BNM)3HW
            id_cond = self.aug_module.augment_one_img(id_cond, only_geo=True)
            id_cond = id_cond.unflatten(0, (B * N, M)).flatten(1, 2)  # (BN)(M3)HW
        id_cond_bkp = id_cond.unflatten(1, (M, 3)).unflatten(0, (B, N))

        # get view embeddings
        view_emb = None
        if self.view_pos_emb is not None:
            view_emb = [
                self.view_pos_emb[i][None].expand(B, -1, -1).flatten(0, 1)[:, None, :]
                for i in range(self.num_layers)
            ]

        # encode input
        st_feats, st_enc_skips = self.st_cnn(x, view_emb=view_emb)

        # encode conditioning and concate
        tex_inp = id_cond
        if self.inp_img_in_cond:
            tex_inp = torch.cat([tex_inp, x], 1)
        _, tex_skips = self.tex_cnn(tex_inp)

        if self.connection_type == "cross_attn":
            skips = st_enc_skips
            context = tex_skips
        else:
            context = None
            skips = {}
            for k, v in tex_skips.items():
                # assert self.cond_pos_emb_type == "none"
                # lev = int(k.split('_')[0])
                # if self.cond_pos_emb_type == "learned":
                #     v = v + self.cond_pos_emb[lev][None]  # pylint: disable=unsubscriptable-object
                # elif self.cond_pos_emb_type == "sin":
                #     v = v + self.cond_pos_emb[lev](v)
                skips[k] = torch.cat([v, st_enc_skips[k]], 2)

        # decode rgb
        emb, _ = self.rgb_decoder(st_feats, skips, context=context, view_emb=view_emb)  # (BN)Dpp
        emb = self.rgb_layer(emb)  # (BN)(H/4W/4)(4*4*3)
        ps = self.patch_size
        emb = emb.unflatten(1, (H // ps, W // ps)).permute(0, 3, 1, 2)  # (BN)(4*4*3)(H/4)(H/4)
        pred_rgb = nn.functional.pixel_shuffle(emb, ps)  # (BN)3HW

        pred_rgb = (pred_rgb + 1) * 128
        pred_rgb = pred_rgb.unflatten(0, (B, N))  # BN3HW
        return pred_rgb, id_cond_bkp
