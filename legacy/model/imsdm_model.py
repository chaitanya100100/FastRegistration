import math
import torch
import torch.nn as nn
import einops
import timm
from ..keypoint import KeypointHandler
from .core import Block, DecBlock
from ..utils import gradient_postprocess


def shufflerow(tensor, axis):
    # get permutation indices
    row_perm = torch.rand(tensor.shape[: axis + 1], device=tensor.device).argsort(axis)
    for _ in range(tensor.ndim - axis - 1):
        row_perm.unsqueeze_(-1)
    # reformat this for the gather operation
    row_perm = row_perm.repeat(*[1 for _ in range(axis + 1)], *(tensor.shape[axis + 1 :]))
    return row_perm


class ImageSDMTransformer(nn.Module):
    def __init__(self, num_params, kp_handler: KeypointHandler, **kwargs) -> None:
        super().__init__()
        self.kp_handler = kp_handler
        self.num_params = num_params
        self.num_kps = kwargs.get('num_kp')
        self.num_views = len(kwargs.get('cameras'))
        self.render_size = kwargs.get("render_size")

        model_kwargs = kwargs.get('model_kwargs_v1')
        self.tsfm_type = model_kwargs.get('tsfm_type')
        assert self.tsfm_type in ['img', 'patch_cls', 'allpatch']
        self.dim = model_kwargs.get('dim')
        self.num_enc_layers = model_kwargs.get('num_enc_layers')
        self.num_dec_layers = model_kwargs.get('num_dec_layers')
        self.num_heads = model_kwargs.get('num_heads')
        self.cnn_channels = model_kwargs.get('cnn_channels')
        self.vit = model_kwargs.get('vit')
        self.pos_emb_once = model_kwargs.get('pos_emb_once')
        self.use_view_pos_emb = model_kwargs.get('use_view_pos_emb')
        assert self.pos_emb_once in [True, False]
        assert self.use_view_pos_emb in [True, False]

        N, P, D = self.num_views, self.num_params, self.dim

        data_feature_type = kwargs.get("data_feature_type")
        in_chans = {
            'rimg_img': 6,
            'res_rimg_img': 3,
            'ir_img': 4,
            'ir3_img': 6,
            'stimg_img': 6,
            'stimg_img_ir': 7,
        }[data_feature_type]
        # self.resnet = timm.create_model(cnn_backbone, pretrained=False, in_chans=in_chans)
        if self.vit:
            dsr = 16
            assert self.render_size[0] == self.render_size[1]
            assert self.render_size[0] % dsr == 0
            self.patch_emb = nn.Conv2d(in_chans, self.cnn_channels[-1], kernel_size=dsr, stride=dsr)
            raise AttributeError
        else:
            self.resnet = timm.create_model(
                'resnetv2_50d_gn', pretrained=False, in_chans=in_chans, channels=self.cnn_channels
            )
            self.resnet.head.fc = torch.nn.Identity()
            dsr = 32
        cnn_feat_dim = self.cnn_channels[-1]

        self.dfeat_emb = nn.Linear(cnn_feat_dim, D)
        self.sfeat_emb = nn.Linear(P, D)
        self.gfeat_emb = nn.Linear(2 * P, D)
        self.comb_emb = nn.Linear(3 * D, D)

        self.view_pos_emb = nn.Parameter(torch.randn(N, D) * 0.02)
        if self.tsfm_type in ['patch_cls', 'allpatch']:
            np = math.ceil(self.render_size[0] / dsr) * math.ceil(self.render_size[1] / dsr)
            self.patch_pos_emb = nn.Parameter(torch.randn(np, D) * 0.02)
            if self.tsfm_type == 'patch_cls':
                self.cls_token = nn.Parameter(torch.randn(D) * 1.0e-6)
                assert self.num_enc_layers % 2 == 0
                assert self.num_dec_layers % 2 == 0

        self.enc_layers = nn.ModuleList(
            [Block(D, self.num_heads, mlp_ratio=2) for _ in range(self.num_enc_layers)]
        )
        self.dec_layers = nn.ModuleList(
            [DecBlock(D, self.num_heads, mlp_ratio=2) for _ in range(self.num_dec_layers)]
        )

        self.out_layer = nn.Linear(D, P)
        # We want to output small updates in early training stage to get meaningful
        # gradients. Large noisy output will be very far from the ground truth and
        # can cause explosion. This significantly helped in early training.
        torch.nn.init.uniform_(self.out_layer.weight, -0.01, 0.01)
        torch.nn.init.zeros_(self.out_layer.bias)

    def forward(self, xs, iter_idx):
        """
        If we have N views and HxWxD features per view,
        (1) 'img': we pool each HxWxD feature into a D-dim feature and transformer works on N tokens.
        (2) 'allpatch': transformer works on total NxHxW tokens simulteneously.
        (3) 'patch_cls': there is one extra pivot token per view i.e. (HxW+1)xD like in ViT's cls token.
                  Transformer layers alternate between attending (HxW+1) tokens of each view
                  and N pivot tokens across views.
        """
        if self.tsfm_type == 'img':
            return self.forward_img_transformer(xs, iter_idx)
        elif self.tsfm_type == 'patch_cls':
            return self.forward_patch_cls_transformer(xs, iter_idx)
        elif self.tsfm_type == 'allpatch':
            return self.forward_allpatch_transformer(xs, iter_idx)

    def forward_patch_cls_transformer(self, xs, iter_idx):
        dfeat, sfeat, gfeat = xs
        gfeat = gradient_postprocess(gfeat)

        N, P, D = self.num_views, self.num_params, self.dim
        B = dfeat.shape[0]
        H, W = dfeat.shape[-2:]

        # dfeat BN6HW
        dfeat = dfeat.view(B * N, -1, H, W)
        dfeat = self.resnet.forward_features(dfeat)  # (BN)D'pp
        dfeat = self.dfeat_emb(dfeat.permute(0, 2, 3, 1))  # (BN)ppD
        dfeat = dfeat.view(B, N, -1, D)  # BNMD
        M = dfeat.shape[2]  # M=p*p

        sfeat = self.sfeat_emb(sfeat.view(B, P))  # BD
        gfeat = self.gfeat_emb(gfeat.view(B, 2 * P))  # BD

        emb = torch.cat(
            [
                dfeat,
                sfeat[:, None, None, :].repeat(1, N, M, 1),
                gfeat[:, None, None, :].repeat(1, N, M, 1),
            ],
            -1,
        )
        emb = self.comb_emb(emb)  # BNMD
        cls = self.cls_token[None, None, None, :].repeat(B, N, 1, 1)  # BN1D

        patch_pos_emb = self.patch_pos_emb[None, None, :, :]  # 11MD
        view_pos_emb = self.view_pos_emb[None, :, None, :]  # 1N1D
        if not self.use_view_pos_emb:
            view_pos_emb = 0

        for i in range(self.num_enc_layers // 2):
            cls_emb = torch.cat([cls, emb + patch_pos_emb], -2)  # BN(M+1)D
            cls_emb = einops.rearrange(cls_emb, 'b n k d -> (b n) k d')
            cls_emb = self.enc_layers[2 * i](cls_emb)
            cls_emb = einops.rearrange(cls_emb, '(b n) k d -> b n k d', b=B)

            cls = cls_emb[:, :, :1, :]  # BN1D
            emb = cls_emb[:, :, 1:, :]  # BNMD

            cls = cls + view_pos_emb  # BN1D
            cls = self.enc_layers[2 * i + 1](cls[:, :, 0, :])[:, :, None, :]  # BN1D

            if self.pos_emb_once and i == 0:
                patch_pos_emb = 0
                view_pos_emb = 0

        cls = cls[:, :, 0, :]  # BND
        emb = sfeat[:, None, :]  # B1D
        for i in range(self.num_dec_layers):
            emb = self.dec_layers[i](emb, cls)

        return self.out_layer(emb.squeeze())

    def forward_allpatch_transformer(self, xs, iter_idx):
        dfeat, sfeat, gfeat = xs
        gfeat = gradient_postprocess(gfeat)

        N, P, D = self.num_views, self.num_params, self.dim
        B = dfeat.shape[0]
        H, W = dfeat.shape[-2:]

        # dfeat BN6HW
        dfeat = dfeat.view(B * N, -1, H, W)
        if self.vit:
            dfeat = self.patch_emb(dfeat)
        else:
            dfeat = self.resnet.forward_features(dfeat)  # (BN)D'pp
        dfeat = self.dfeat_emb(dfeat.permute(0, 2, 3, 1))  # (BN)ppD
        dfeat = dfeat.view(B, N, -1, D)  # BNMD
        M = dfeat.shape[2]  # M=p*p

        sfeat = self.sfeat_emb(sfeat.view(B, P))  # BD
        gfeat = self.gfeat_emb(gfeat.view(B, 2 * P))  # BD

        emb = torch.cat(
            [
                dfeat,
                sfeat[:, None, None, :].repeat(1, N, M, 1),
                gfeat[:, None, None, :].repeat(1, N, M, 1),
            ],
            -1,
        )
        emb = self.comb_emb(emb)  # BNMD
        patch_pos_emb = self.patch_pos_emb[None, None, :, :]  # 11MD
        view_pos_emb = self.view_pos_emb[None, :, None, :]  # 1N1D
        if not self.use_view_pos_emb:
            view_pos_emb = 0

        for i in range(self.num_enc_layers):
            emb = einops.rearrange(emb + patch_pos_emb + view_pos_emb, 'b n k d -> (b n) k d')
            emb = self.enc_layers[i](emb)
            emb = einops.rearrange(emb, '(b n) k d -> b n k d', b=B)
            if self.pos_emb_once and i == 0:
                patch_pos_emb = 0
                view_pos_emb = 0

        cemb = einops.rearrange(emb, 'b n k d -> b (n k) d')
        emb = sfeat[:, None, :]  # B1D
        for i in range(self.num_dec_layers):
            emb = self.dec_layers[i](emb, cemb)

        return self.out_layer(emb.squeeze())

    def forward_img_transformer(self, xs, iter_idx):
        dfeat, sfeat, gfeat = xs
        gfeat = gradient_postprocess(gfeat)

        N, P, D = self.num_views, self.num_params, self.dim
        B = dfeat.shape[0]
        H, W = dfeat.shape[-2:]

        # dfeat BN6HW
        dfeat = dfeat.view(B * N, -1, H, W)
        dfeat = self.resnet(dfeat)  # (BN)512
        dfeat = self.dfeat_emb(dfeat).view(B, N, D)  # BND

        sfeat = self.sfeat_emb(sfeat.view(B, P))  # BD
        gfeat = self.gfeat_emb(gfeat.view(B, 2 * P))  # BD

        emb = torch.cat(
            [
                dfeat,
                sfeat[:, None, :].repeat(1, N, 1),
                gfeat[:, None, :].repeat(1, N, 1),
            ],
            -1,
        )
        emb = self.comb_emb(emb)  # BND

        view_pos_emb = self.view_pos_emb[None, :, :]  # 1ND
        if not self.use_view_pos_emb:
            view_pos_emb = 0

        for i in range(self.num_enc_layers):
            emb = self.enc_layers[i](emb + view_pos_emb)
            if self.pos_emb_once and i == 0:
                view_pos_emb = 0

        cemb = emb  # BND
        emb = sfeat[:, None, :]  # B1D
        for i in range(self.num_dec_layers):
            emb = self.dec_layers[i](emb, cemb)

        return self.out_layer(emb.squeeze())
