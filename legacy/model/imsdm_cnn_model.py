"""
SDM model code without transformer. Used for ablation in paper. Not useful.
"""

import torch
import torch.nn as nn
import timm
from ..keypoint import KeypointHandler
from ..utils import gradient_postprocess
from .core import MyResBlock, my_groupnorm


class ImageSDMCNNMLP(nn.Module):
    def __init__(self, num_params, kp_handler: KeypointHandler, **kwargs) -> None:
        super().__init__()
        self.kp_handler = kp_handler
        self.num_params = num_params
        self.num_kps = kwargs.get('num_kp')
        self.num_views = len(kwargs.get('cameras'))
        self.render_size = kwargs.get("render_size")
        self.cnn_channels = [128, 256, 512, 1024]

        N, P = self.num_views, self.num_params
        D = self.cnn_channels[-1]
        self.dim = D

        data_feature_type = kwargs.get("data_feature_type")
        in_chans = {
            'rimg_img': 6,
            'res_rimg_img': 3,
            'ir_img': 4,
            'ir3_img': 6,
            'stimg_img': 6,
            'stimg_img_ir': 7,
        }[data_feature_type]

        self.resnet = timm.create_model(
            'resnetv2_50d_gn', pretrained=False, in_chans=in_chans, channels=self.cnn_channels
        )
        self.resnet.head.fc = torch.nn.Identity()

        def get_resblock(D1, D2):
            return nn.Sequential(nn.Linear(D1, D2), MyResBlock(D2, nn.GELU, my_groupnorm, None))

        self.dfeat_emb = get_resblock(D, D)
        self.sfeat_emb = nn.Linear(P, D)
        self.gfeat_emb = nn.Linear(2 * P, D)
        self.comb_emb = nn.Sequential(get_resblock(3 * D, D))

        self.pool = False
        if self.pool:
            self.fuse_emb = nn.Sequential(
                get_resblock(N * D, D), get_resblock(D, D), get_resblock(D, D)
            )
        else:
            self.fuse_emb = nn.Sequential(
                get_resblock(16 * N * D, D), get_resblock(D, D), get_resblock(D, D)
            )
        self.out_layer = nn.Linear(D, P)
        # We want to output small updates in early training stage to get meaningful
        # gradients. Large noisy output will be very far from the ground truth and
        # can cause explosion. This significantly helped in early training.
        torch.nn.init.uniform_(self.out_layer.weight, -0.01, 0.01)
        torch.nn.init.zeros_(self.out_layer.bias)

    def forward(self, xs, iter_idx):
        dfeat, sfeat, gfeat = xs
        gfeat = gradient_postprocess(gfeat)

        N, P, D = self.num_views, self.num_params, self.dim
        B = dfeat.shape[0]

        # dfeat BN6HW
        dfeat = self.resnet.forward_features(dfeat.flatten(0, 1))  # (BN)D'pp
        if self.pool:
            dfeat = dfeat.mean((-1, -2))  # (BN)D'
            dfeat = self.dfeat_emb(dfeat)  # (BN)D
            dfeat = dfeat.view(B, N, D)  # BND

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
            emb = self.comb_emb(emb.flatten(0, 1)).unflatten(0, (B, N))  # BND
            emb = self.fuse_emb(emb.flatten(1, 2))  # BD
        else:
            M = dfeat.shape[-1] * dfeat.shape[-2]
            dfeat = dfeat.view(B, N, D, -1).transpose(-1, -2).flatten(0, 2)  # (BNM)D
            dfeat = self.dfeat_emb(dfeat)  # (BNM)D

            sfeat = self.sfeat_emb(sfeat.view(B, P))  # BD
            gfeat = self.gfeat_emb(gfeat.view(B, 2 * P))  # BD

            emb = torch.cat(
                [
                    dfeat,
                    sfeat[:, None, None, :].expand(-1, N, M, -1).flatten(0, 2),
                    gfeat[:, None, None, :].expand(-1, N, M, -1).flatten(0, 2),
                ],
                -1,
            )
            emb = self.comb_emb(emb)  # (BNM)D
            emb = emb.unflatten(0, (B, N, M)).flatten(1, 3)  # B(NMD)
            emb = self.fuse_emb(emb)  # BD

        return self.out_layer(emb)
