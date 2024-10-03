import torch
import torch.nn as nn
import einops
from model.core import (
    MyResMLP,
    SinusoidalPositionEmbeddings,
    get_activation,
    get_act_normalization,
    Block,
    DecBlock,
)
from ..keypoint import KeypointHandler
from ..utils import gradient_postprocess


def get_guiding_dim(guiding_feature_type, K, N, P):
    # K, N, P are num_kp, num_views, num_pred_params
    return {
        "grad": P,
        "grad_processed": P * 2,
        "jac": N * K * 2 * P,
        "jac_processed": N * K * 2 * P * 2,
    }[guiding_feature_type]


class MyResMLPLateFusion(nn.Module):
    def __init__(self, num_pred_params, **kwargs):
        super().__init__()
        print(kwargs)
        P = num_pred_params
        N = kwargs.get('num_views')
        K = kwargs.get('num_reduced_kp')
        guiding_feature_type = kwargs.get('guiding_feature_type')
        self.guiding_feature_type = guiding_feature_type

        model_kwargs = kwargs.get('mrmlp_kwargs')
        self.view_handling = model_kwargs.get('view_handling')
        hidden_dim = model_kwargs.get('hidden_dim')
        num_hiddens = model_kwargs.get('num_hiddens')
        preproc_hidden_dim = model_kwargs.get('preproc_hidden_dim')
        preproc_num_hiddens = model_kwargs.get('preproc_num_hiddens')
        activation = get_activation(model_kwargs.get('activation'))
        act_normalization = get_act_normalization(model_kwargs.get('act_normalization'))
        self.ss_type = model_kwargs.get('scale_shift_type')
        assert self.ss_type in ["none", "time", "res", "loss", "total_loss"]
        self.ss_type = None if self.ss_type == "none" else self.ss_type

        time_dim = 512 if self.ss_type is not None else None

        print("View handling is ", self.view_handling)
        self.num_views = N
        if self.view_handling == "stacked":
            orig_input_dims = [N * K * 2, P, get_guiding_dim(guiding_feature_type, K, N, P)]
            input_dims = orig_input_dims
        elif self.view_handling == "sharedview_onehot":
            orig_input_dims = [K * 2, P, get_guiding_dim(guiding_feature_type, K, 1, P), N]
            input_dims = orig_input_dims
        elif self.view_handling == "sharedview_posemb":
            orig_input_dims = [K * 2, P, get_guiding_dim(guiding_feature_type, K, 1, P)]
            self.inp_emb = nn.ModuleList(
                [nn.Linear(id_, preproc_hidden_dim) for id_ in orig_input_dims]
            )
            self.view_emb = nn.Parameter(torch.randn(N, preproc_hidden_dim) * 0.02)
            input_dims = [preproc_hidden_dim] * 3
        else:
            raise AttributeError

        self.preprocessors = []
        proproc_kwargs = dict(
            output_dim=preproc_hidden_dim,
            hidden_dim=preproc_hidden_dim,
            activation=activation,
            num_hiddens=preproc_num_hiddens,
            act_normalization=act_normalization,
            time_dim=time_dim,
        )
        self.preprocessors = nn.ModuleList([MyResMLP(id, **proproc_kwargs) for id in input_dims])

        self.block = MyResMLP(
            preproc_hidden_dim * len(input_dims),
            num_pred_params,
            hidden_dim,
            activation=activation,
            num_hiddens=num_hiddens,
            act_normalization=act_normalization,
            time_dim=time_dim,
        )

        self.ss_block = None
        self.ss_idx = 0
        if self.ss_type == "time":
            self.ss_block = nn.Sequential(
                SinusoidalPositionEmbeddings(time_dim),
                nn.Linear(time_dim, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, time_dim),
            )
        elif self.ss_type in ["res", "loss", "total_loss"]:
            ginpd = orig_input_dims[self.ss_idx]
            if self.ss_type == "loss":
                ginpd //= 2
            if self.ss_type == "total_loss":
                ginpd = 1
            print(f"{self.ss_type} scale shift with dim", ginpd)
            self.ss_block = nn.Sequential(
                nn.Linear(ginpd, hidden_dim),
                activation(),
                nn.Linear(hidden_dim, time_dim),
            )

    def forward(self, xs, iter_idx):
        if 'processed' in self.guiding_feature_type:
            dfeat, sfeat, gfeat = xs
            gfeat = gradient_postprocess(gfeat)
            xs = [dfeat, sfeat, gfeat]

        t = None
        if self.ss_type is not None:
            if self.ss_type == "time":
                t = torch.ones(xs[0].shape[0], device=xs[0].device) * iter_idx
            elif self.ss_type == "res":
                t = xs[self.ss_idx]
            elif self.ss_type == "loss":
                t = xs[self.ss_idx].view(t.shape[0], -1, 2).square().sum(-1).sqrt()
            elif self.ss_type == "total_loss":
                t = xs[self.ss_idx].view(t.shape[0], -1, 2).square().sum(-1).sqrt()
                t = t.mean(-1)[:, None]
            t = self.ss_block(t)

        if self.iterwise_normalizer is not None:
            xs = [fn(x) for x, fn in zip(xs, self.iterwise_normalizer[iter_idx])]

        if self.view_handling == "stacked":
            px = torch.cat([prepro(x, t) for prepro, x in zip(self.preprocessors, xs)], 1)
            out = self.block(px, t)
        elif self.view_handling in ["sharedview_onehot", "sharedview_posemb"]:
            dfeat, sfeat, gfeat = xs
            B = dfeat.shape[0]
            N = self.num_views

            dfeat = dfeat.view(B * N, -1)  # (BN)a
            sfeat = sfeat[:, None].expand(-1, N, -1).flatten(0, 1)  # (BN)b

            if self.guiding_feature_type in ["jac", "jac_processed"]:
                gfeat = gfeat.view(B * N, -1)
            else:
                gfeat = gfeat[:, None].expand(-1, N, -1).flatten(0, 1)

            if self.view_handling == "sharedview_onehot":
                voh = nn.functional.one_hot(torch.arange(N, device=dfeat.device)).float()  # NN
                voh = voh[None].expand(B, -1, -1).flatten(0, 1)  # (BN)N
                xs = [dfeat, sfeat, gfeat, voh]
            elif self.view_handling == "sharedview_posemb":
                xs = [dfeat, sfeat, gfeat]
                xs = [emb(x) for emb, x in zip(self.inp_emb, xs)]  # a list of (BN)D
                xs = [(x.view(B, N, -1) + self.view_emb[None]).view(B * N, -1) for x in xs]

            px = torch.cat([prepro(x, t) for prepro, x in zip(self.preprocessors, xs)], 1)
            out = self.block(px, t)
            out = out.view(B, N, -1).mean(1)

        return out


class TransformerSDM(nn.Module):
    def __init__(self, num_params, kp_handler: KeypointHandler, **kwargs) -> None:
        super().__init__()
        self.kp_handler = kp_handler
        self.num_params = num_params
        self.num_kps = kwargs.get('num_kp')
        self.num_views = kwargs.get('num_views')
        self.guiding_feature_type = kwargs.get("guiding_feature_type")
        assert self.guiding_feature_type in ["grad_processed", "jac_processed"]

        model_kwargs = kwargs.get('tsfm_kwargs')
        self.dim = model_kwargs.get('dim')
        self.num_enc_layers = model_kwargs.get('num_enc_layers')
        self.num_dec_layers = model_kwargs.get('num_dec_layers')
        self.num_heads = model_kwargs.get('num_heads')
        self.use_kp_subset = model_kwargs.get('use_kp_subset')

        K, N, P, D = self.num_kps, self.num_views, self.num_params, self.dim
        self.dfeat_emb = nn.Linear(2, D)
        self.sfeat_emb = nn.Linear(P, D)
        self.gfeat_emb = nn.Linear(2 * P * 2, D)
        self.comb_emb = nn.Linear(3 * D, D)

        self.kp_pos_emb = nn.Parameter(torch.randn(K, D) * 0.02)
        self.view_pos_emb = nn.Parameter(torch.randn(N, D) * 0.02)

        self.enc_layers = []
        for _ in range(self.num_enc_layers):
            self.enc_layers.append(Block(D, self.num_heads, mlp_ratio=1))
            self.enc_layers.append(Block(D, self.num_heads, mlp_ratio=1))
        self.enc_layers = nn.ModuleList(self.enc_layers)

        self.dec_layers = []
        for _ in range(self.num_dec_layers):
            self.dec_layers.append(DecBlock(D, self.num_heads, mlp_ratio=1))
            self.dec_layers.append(DecBlock(D, self.num_heads, mlp_ratio=1))
        self.dec_layers = nn.ModuleList(self.dec_layers)
        self.out_layer = nn.Linear(D, P)
        # We want to output small updates in early training stage to get meaningful
        # gradients. Large noisy output will be very far from the ground truth and
        # can cause explosion. This significantly helped in early training.
        torch.nn.init.uniform_(self.out_layer.weight, -0.01, 0.01)
        torch.nn.init.zeros_(self.out_layer.bias)

    def forward(self, xs, iter_idx):
        dfeat, sfeat, gfeat = xs
        if 'processed' in self.guiding_feature_type:
            gfeat = gradient_postprocess(gfeat)
        K, N, P, D = self.num_kps, self.num_views, self.num_params, self.dim
        B = dfeat.shape[0]

        dfeat = dfeat.view(B, N, K, 2)
        sfeat = sfeat.view(B, P)
        if self.guiding_feature_type == "grad_processed":
            gfeat = (
                torch.cat([gfeat, gfeat], -1)[:, None, None, :].expand(-1, N, K, -1).contiguous()
            )  # BNK(4P)
        gfeat = gfeat.view(B, N, K, 2 * P * 2)

        if self.iterwise_normalizer is not None:
            dfeat = self.iterwise_normalizer[iter_idx][0](dfeat.view(-1, 2)).view(dfeat.shape)
            sfeat = self.iterwise_normalizer[iter_idx][1](sfeat)
            gfeat = self.iterwise_normalizer[iter_idx][2](gfeat.view(-1, 2 * P * 2)).view(
                gfeat.shape
            )

        sfeat = self.sfeat_emb(sfeat)  # BD
        dfeat = self.dfeat_emb(dfeat.view(-1, 2)).view(B, N, K, D)  # BNKD
        gfeat = self.gfeat_emb(gfeat.view(-1, 2 * P * 2)).view(B, N, K, D)  # BNKD

        emb = self.comb_emb(
            torch.cat([dfeat, sfeat[:, None, None, :].repeat(1, N, K, 1), gfeat], -1)
        )  # BNKD

        if not self.use_kp_subset:
            kp_pos_emb = self.kp_pos_emb[None, None, :, :]  # 11KD
            view_pos_emb = self.view_pos_emb[None, :, None, :]  # 1N1D
        else:
            kp_pos_emb = self.kp_pos_emb[None, None, ...].expand(-1, N, -1, -1)  # 1NKD
            kp_pos_emb = self.kp_handler.select_subset(kp_pos_emb)  # 1NK'D
            view_pos_emb = self.view_pos_emb[None, :, None, :].expand(-1, -1, K, -1)  # 1NKD
            view_pos_emb = self.kp_handler.select_subset(view_pos_emb)  # 1NK'D
            emb = self.kp_handler.select_subset(emb)  # BNK'D

        for i in range(self.num_enc_layers):
            emb = einops.rearrange(emb + kp_pos_emb, 'b n k d -> (b n) k d')
            emb = self.enc_layers[2 * i](emb)
            emb = einops.rearrange(emb, '(b n) k d -> b n k d', b=B)
            emb = einops.rearrange(emb + view_pos_emb, 'b n k d -> (b k) n d')
            emb = self.enc_layers[2 * i + 1](emb)
            emb = einops.rearrange(emb, '(b k) n d -> b n k d', b=B)

        cembk = einops.rearrange(emb, 'b n k d -> (b n) k d')  # (BN)KD
        cembn = einops.rearrange(emb, 'b n k d -> (b k) n d')  # (BK)ND
        emb = sfeat[:, None, :]  # B1D
        for i in range(self.num_dec_layers):
            emb = emb[:, None].expand(-1, N, -1, -1).flatten(0, 1)  # (BN)1D
            emb = self.dec_layers[2 * i](emb, cembk)
            emb = emb.view(B, N, 1, D).mean(1)  # B1D

            kvar = cembk.shape[1]
            emb = emb[:, None].expand(-1, K, -1, -1).flatten(0, 1)  # (BK)1D
            emb = self.dec_layers[2 * i + 1](emb, cembn)
            emb = emb.view(B, kvar, 1, D).mean(1)  # B1D

        return self.out_layer(emb.squeeze())


class TransformerSDMTwoModels(nn.Module):
    def __init__(self, num_params, kp_handler: KeypointHandler, **kwargs) -> None:
        super().__init__()
        self.mod1 = TransformerSDM(num_params, kp_handler, **kwargs)
        self.mod2 = TransformerSDM(num_params, kp_handler, **kwargs)

    def forward(self, xs, iter_idx):
        if iter_idx == 0:
            return self.mod1(xs, iter_idx)
        else:
            return self.mod2(xs, iter_idx)
