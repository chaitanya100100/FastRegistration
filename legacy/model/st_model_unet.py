# import math
import torch

# import torchvision
from torch import nn
from randaugment import MyAugment
from .unet_core import (
    UNetDecoder,
    UNetEncoder,
    UNet,
    ResnetBlock,
    # MyConvNeXtBlock,
    Downsample,
    Upsample,
    ImageAttentionGeneral,
    ImageTBlockGeneral,
)

# from .core import TBlockGeneral
# from .pos_encoding import PositionalEncodingPermute2D
from .core import flip_u_every_other
from .core import SASAGeneral

# from .core import mySequential


class STCnnUVMap(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        model_kwargs = kwargs.get('model_uv_kwargs')
        self.uv_model = UNet(
            inp_chan=1,
            out_chan=2 + 1,
            first_dim=model_kwargs['first_dim'],
            num_levels=model_kwargs['num_levels'],
            attn_start_level=model_kwargs['attn_start_level'],
            tblock=model_kwargs['tblock'],
            dropout=model_kwargs['dropout'],
        )
        self.cameras = (
            model_kwargs['cameras'] if model_kwargs['cameras'] is not None else kwargs['cameras']
        )

    def forward(self, x, view_idx):
        B, N, _, _, _ = x.shape
        x = x.flatten(0, 1)

        pred_uvm = self.uv_model.forward(x)  # (BN)Dpp
        pred_uvm = pred_uvm.unflatten(0, (B, N))  # BN(2+1)HW

        pred_uv, pred_mask_logits = pred_uvm[:, :, :2], pred_uvm[:, :, 2:]  # BN2HW, BN1HW
        pred_uv = flip_u_every_other(pred_uv)
        return pred_uv, pred_mask_logits


class STCnnRGB(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.tex_inp_size = kwargs.get('tex_inp_size')
        self.cond_type = kwargs.get('cond_type')
        self.inp_img_in_cond = kwargs.get('inp_img_in_cond')
        self.num_cond_images = len(kwargs.get('cond_frame_num'))
        num_views = len(kwargs['cameras'])

        model_kwargs = kwargs.get('cnn_rgb_kwargs')
        self.first_dim = model_kwargs['first_dim']
        self.num_levels = model_kwargs['num_levels']
        self.attn_start_level = model_kwargs['attn_start_level']
        dropout = model_kwargs['dropout']
        tblock = model_kwargs['tblock']
        view_emb = model_kwargs['view_emb']
        default_norm = model_kwargs['default_norm']
        self.use_cond_dec = model_kwargs['use_cond_dec']
        self.tex_hmc_join_type = model_kwargs['tex_hmc_join_type']
        assert self.tex_hmc_join_type in ['concat', 'sasa']

        use_attn_cls = {False: ImageAttentionGeneral, True: ImageTBlockGeneral}[tblock]
        conv_block = ResnetBlock

        conduv = 'conduv' in self.cond_type
        M = self.num_cond_images
        self.tex_cnn = UNetEncoder(
            in_channels=(3 + 3 * int(conduv)) * M + int(self.inp_img_in_cond),
            first_dim=self.first_dim,
            num_levels=self.num_levels,
            attn_start_level=self.attn_start_level,
            dropout=dropout,
            use_attn_cls=use_attn_cls,
            norm='default' if default_norm else 'group',
        )
        if self.use_cond_dec:
            self.tex_decoder = UNetDecoder(
                self.first_dim,
                num_levels=self.num_levels,
                attn_start_level=self.attn_start_level,
                context_dim=None,
                dropout=dropout,
                use_attn_cls=use_attn_cls,
                norm='default' if default_norm else 'group',
            )

        self.st_cnn = UNetEncoder(
            in_channels=1,
            first_dim=self.first_dim,
            num_levels=self.num_levels,
            attn_start_level=self.attn_start_level,
            dropout=dropout,
            use_attn_cls=use_attn_cls,
            norm='default' if default_norm else 'instance',
        )

        last_dim = self.first_dim * (2**self.num_levels)
        self.inp_mid = nn.Sequential(
            Downsample(last_dim // 2, last_dim),
            conv_block(last_dim, last_dim, dropout=dropout),
            use_attn_cls(last_dim, num_heads=last_dim // self.first_dim, proj_drop=dropout)
            if self.attn_start_level <= self.num_levels
            else nn.Identity(),
            conv_block(last_dim, last_dim, dropout=dropout),
            Upsample(last_dim, last_dim // 2),
        )

        self.view_pos_emb = None
        if view_emb:
            con_dim = {i: self.first_dim * (2**i) for i in range(self.num_levels)}

            def get_view_emb():
                return nn.ParameterList(
                    [
                        nn.Parameter(torch.randn(num_views, con_dim[i]) * 0.02)
                        for i in range(self.num_levels)
                    ]
                )

            self.view_pos_emb = get_view_emb()
            self.view_pos_emb_1 = get_view_emb()
            self.view_pos_emb_2 = get_view_emb()
            self.view_pos_emb_3 = get_view_emb()

        if self.tex_hmc_join_type == 'sasa':
            con_dim = {i: self.first_dim * (2**i) for i in range(self.num_levels)}
            self.sasa_layers = nn.ModuleDict(
                {
                    f"{i}_{j}": SASAGeneral(con_dim[i], kernel_size=5, num_heads=con_dim[i] // 16)
                    # f"{i}_{j}": mySequential(
                    #     *[
                    #         SASAGeneral(con_dim[i], kernel_size=5, num_heads=con_dim[i] // 16)
                    #         for _ in range(2)
                    #     ]
                    # )
                    for i in range(self.num_levels)
                    for j in range(2)
                }
            )

        self.rgb_decoder = UNetDecoder(
            self.first_dim,
            num_levels=self.num_levels,
            attn_start_level=self.attn_start_level,
            context_dim=None,
            dropout=dropout,
            use_attn_cls=use_attn_cls,
            num_skips=2,
            norm='default' if default_norm else 'group',
        )
        self.rgb_layer = nn.Conv2d(self.first_dim, 3, kernel_size=1)

        print("number of parameters", sum(p.numel() for p in self.tex_cnn.parameters()))
        print("number of parameters", sum(p.numel() for p in self.st_cnn.parameters()))
        print("number of parameters", sum(p.numel() for p in self.inp_mid.parameters()))
        print("number of parameters", sum(p.numel() for p in self.rgb_decoder.parameters()))

        self.aug_module = None
        image_aug = kwargs.get("image_aug")
        if image_aug != "none":
            print("Using image augmentation", image_aug)
            aug_scale = int(image_aug.split("_")[1])
            self.aug_module = MyAugment(aug_scale)

    def forward(self, x, id_cond, uv_outputs):
        # x: B x N x C x H x W
        # id_cond: B x N x M x C x H x W
        # uv: B x N x 2 x H x W

        B, N, _, _, _ = x.shape
        M = id_cond.shape[2]
        x = x.flatten(0, 1)
        id_cond = id_cond.flatten(0, 1).flatten(1, 2)  # (BN)(M3)HW

        if '_uv_' in self.cond_type:
            raise AttributeError("uv conditioning not supported as of now")
            uv = uv_outputs['pred_uv'].flatten(0, 1)  # (BN)2HW

            id_cond = nn.functional.grid_sample(
                id_cond, uv.permute(0, 2, 3, 1), align_corners=False
            )  # (BN)(M3)HW with spatially aligned with image
            mask = (uv_outputs['pred_mask_logits'] > 0).float().flatten(0, 1)
            id_cond = mask * id_cond  # (BN)(M3)HW

        # if self.aug_module is not None and self.training:
        #     id_cond = id_cond.unflatten(1, (M, 3)).flatten(0, 1)  # (BNM)3HW
        #     id_cond = self.aug_module.augment_one_img(id_cond, only_geo=True)
        #     id_cond = id_cond.unflatten(0, (B * N, M)).flatten(1, 2)  # (BN)(M3)HW

        # id_cond_bkp = id_cond.unflatten(1, (M, 3)).unflatten(0, (B, N))
        id_cond_bkp = id_cond.unflatten(1, (M, -1)).unflatten(0, (B, N))[:, :, :, :3]

        # get view embeddings
        if self.view_pos_emb is not None:

            def get_view_emb(bla):
                return [
                    bla[i][None].expand(B, -1, -1).flatten(0, 1)[:, :, None, None]
                    for i in range(self.num_levels)
                ]

            view_emb = get_view_emb(self.view_pos_emb)
            view_emb_1 = get_view_emb(self.view_pos_emb_1)
            view_emb_2 = get_view_emb(self.view_pos_emb_2)
            view_emb_3 = get_view_emb(self.view_pos_emb_3)
        else:
            view_emb, view_emb_1, view_emb_2, view_emb_3 = None, None, None, None

        # encode input
        st_feats, st_enc_skips = self.st_cnn(x, view_emb=view_emb)
        st_feats = self.inp_mid(st_feats)

        # encode conditioning and concate
        tex_inp = id_cond
        if self.inp_img_in_cond:
            tex_inp = torch.cat([tex_inp, x], 1)
        tex_feats, tex_skips = self.tex_cnn(tex_inp, view_emb=view_emb_1)
        if self.use_cond_dec:
            tex_feats, tex_skips = self.tex_decoder(
                tex_feats, tex_skips, None, None, view_emb=view_emb_2
            )

        skips = {}
        for k, v in tex_skips.items():
            # lev = int(k.split('_')[0])
            # if self.cond_pos_emb_type == "learned":
            #     v = v + self.cond_pos_emb[lev][None]  # pylint: disable=unsubscriptable-object
            # elif self.cond_pos_emb_type == "sin":
            #     v = v + self.cond_pos_emb[lev](v)

            if self.tex_hmc_join_type == 'concat':
                skips[k] = torch.cat([v, st_enc_skips[k]], 1)
            elif self.tex_hmc_join_type == 'sasa':
                v = self.sasa_layers[k](st_enc_skips[k], v)
                skips[k] = torch.cat([v, st_enc_skips[k]], 1)
            else:
                raise NotImplementedError

        # decode rgb
        emb, _ = self.rgb_decoder(st_feats, skips, None, None, view_emb=view_emb_3)  # (BN)Dpp
        pred_rgb = self.rgb_layer(emb)  # (BN)3HW
        pred_rgb = (pred_rgb + 1) * 128

        pred_rgb = pred_rgb.unflatten(0, (B, N))  # BN3HW
        return pred_rgb, id_cond_bkp
