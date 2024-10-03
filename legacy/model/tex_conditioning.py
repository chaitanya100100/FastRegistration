from typing import Dict, Sequence
import joblib
import numpy as np
import torch as th

from drtk.renderlayer.renderlayer import RenderLayer as RenderLayer

from utils import (
    load_rosetta_stats,
    gather_camera_params,
    resize2d,
)
from sdm_utils import splitbatch_decoder_forward

# import IPython  # pylint: disable=unused-import

from sdm_utils import State


class TexCond(th.nn.Module):
    def __init__(self, decoder, **kwargs):
        super().__init__()
        self.decoder = decoder
        topology_file = kwargs.get("tex_topology_file")
        cond_expr_code_path = kwargs.get("cond_expr_code")
        cond_frame_num = kwargs.get("cond_frame_num")
        self.tex_inp_size = kwargs.get("tex_inp_size")
        self.render_size = kwargs.get("render_size")
        self.cond_type = kwargs.get("cond_type")
        self.cond_perid_centering = kwargs.get("cond_perid_centering")
        self.cameras = kwargs['cameras']
        self.isMouthView = [('mouth' in c) for c in self.cameras]

        self.topology = typed.load(get_repo_asset_path(topology_file))

        self.front_view_generator = FixedFrontalViewGenerator()
        self.renderer = RenderLayer(
            400,  # these sizes don't matter.
            400,
            self.topology["vt"],
            self.topology["vi"],
            self.topology["vti"],
            boundary_aware=False,
            flip_uvs=False,
        )
        self.renderer_for_unwrap = RenderLayer(
            1024,
            1024,
            self.topology["vt"],
            self.topology["vi"],
            self.topology["vti"],
            boundary_aware=False,
            flip_uvs=False,
        )

        # assigning one texture coordinate to each vertex
        geo_uv = np.zeros((self.topology["v"].shape[0], 2), dtype=np.float32) - 1
        # vt_to_v = th.LongTensor(np.zeros((self.topology["vt"].shape[0],)))
        for i in range(self.topology["vi"].shape[0]):
            this_tri = self.topology["vi"][i]
            this_vti = self.topology["vti"][i]
            for j in range(3):
                vindex = this_tri[j]
                vtindex = this_vti[j]
                if geo_uv[vindex][0] == -1:
                    geo_uv[vindex] = self.topology['vt'][vtindex]
                    # vt_to_v[vtindex] = vindex
        geo_uv[:, 1] = 1 - geo_uv[:, 1]
        self.geo_uv = geo_uv

        # texture index to vertex index
        n_vt = self.renderer.vt.shape[0]
        vi_flat = self.renderer.vi.reshape(-1).cpu()
        vti_flat = self.renderer.vti.reshape(-1).cpu()
        vt_to_v = th.LongTensor(np.zeros((n_vt,)))  # - 1
        for vt_item, v_item in zip(vti_flat, vi_flat):
            vt_to_v[vt_item] = v_item
        self.vt_to_v = vt_to_v

        # To cache conditioning info for each identity
        self.cond_info = {}
        cond_codes = joblib.load(cond_expr_code_path)
        self.gt_expr = -1 in cond_frame_num
        if self.gt_expr:
            cond_frame_num.remove(-1)

        self.sdm_expr = -2 in cond_frame_num
        if self.sdm_expr:
            assert 'sdm' in self.cond_type
            cond_frame_num.remove(-2)

        cond_codes = [cond_codes[i][2] for i in cond_frame_num]
        cond_codes = (
            th.zeros(0, 256)
            if len(cond_codes) == 0
            else th.from_numpy(np.stack(cond_codes)).float()
        )
        self.register_buffer("cond_codes", cond_codes, persistent=False)

        self.register_buffer("dummy_tex", th.zeros(1, 3, 1024, 1024), persistent=False)

        stats = load_rosetta_stats()
        self.rotdim = kwargs.get("rotdim", 6)
        rotkey = 'avatar2hmc_rot6d' if self.rotdim == 6 else 'avatar2hmc_rvec'
        mean_rtvec = th.cat([stats[rotkey]['mean'], stats['avatar2hmc_tvec']['mean']])
        std_rtvec = th.cat([stats[rotkey]['std'], stats['avatar2hmc_tvec']['std']])
        self.register_buffer("mean_rtvec", mean_rtvec, persistent=False)
        self.register_buffer("std_rtvec", std_rtvec, persistent=False)
        if self.cond_perid_centering:
            self.perid_stats = stats['perid_stats']
            mean_expr = self.perid_stats["AAL724"]['mean'][None]
            self.cond_codes = self.cond_codes - mean_expr.to(self.cond_codes.device)
        else:
            raise AttributeError

        u, v = th.meshgrid(th.linspace(-1, 1, 1024), th.linspace(-1, 1, 1024), indexing='xy')
        self.register_buffer("uvcoord", th.stack([u, v, th.ones_like(u)])[None], persistent=False)

    def get_decoder_maps(self, idents: Sequence[str]) -> Dict[str, th.Tensor]:
        # batch['index']['ident']
        mugsy_idents = [self.decoder.ident_str_mapping[ident] for ident in idents]
        id_cond = self.decoder.decoder.get_id_cond(mugsy_idents)
        return id_cond

    def unwrap_texture(self, ident, expr, camera_params):
        B = len(ident)

        with th.no_grad():
            decb = splitbatch_decoder_forward(
                self.decoder, ident, expr, camera_params, onepass=(B < 64)
            )
        # decb, _ = self.decoder.forward(
        #     data={'index': {'ident': ident}},
        #     inputs={
        #         'expression': expr,
        #         'camera_params': camera_params,
        #     },
        #     compute_losses=False,
        # )

        geo = decb['geo_pred'].detach()  # BV3
        imgs = decb['img_pred'].detach()  # B3HW
        masks = decb['mask'].detach()  # B1HW

        if '_uv_' not in self.cond_type and 'conduv' not in self.cond_type:
            return {
                'img': imgs,
                # 'mask': masks,
            }

        h = camera_params.pop("render_h")
        w = camera_params.pop("render_w")
        camera_params.pop("isMouthView", None)
        self.renderer.resize(h, w)
        render_out = self.renderer(
            geo,
            self.uvcoord.expand(B, -1, -1, -1),
            output_filters=["bary_img", "index_img", "v_pix", "render"],
            **camera_params,
        )
        uv_imgs = render_out['render']  # B3HW
        uv_imgs = th.stack([uv_imgs[:, 0], -uv_imgs[:, 1]], 1)  # B2HW

        if '_uv_' not in self.cond_type and 'conduv' in self.cond_type:
            return {
                'img': imgs,
                'mask': masks,
                'uv_img': uv_imgs,
            }

        v2d = render_out["v_pix"][:, :, :2]
        tex = imgs
        device = v2d.device
        v2d_norm = v2d / th.Tensor([tex.shape[3], tex.shape[2]]).to(device)
        vt_new = v2d_norm[:, self.vt_to_v, :]

        g = (
            th.cat((1024 * (th.from_numpy(self.geo_uv) / 1.0 + 0.0), th.ones(7306, 1)), 1)
            .unsqueeze(0)
            .to(device)
        )

        uwrapped_texs = []
        unwrapped_masks = []
        for i in range(B):
            visible_face_indices = set(th.unique(render_out["index_img"][i]).tolist())
            if -1 in visible_face_indices:
                visible_face_indices.remove(-1)
            visible_face_indices = th.LongTensor(list(visible_face_indices))

            self.renderer_for_unwrap.vi = self.renderer.vi[visible_face_indices]
            self.renderer_for_unwrap.vti = self.renderer.vti[visible_face_indices]
            self.renderer_for_unwrap.vt = vt_new[i]

            out = self.renderer_for_unwrap.forward(
                g,
                tex[i : i + 1],
                th.zeros(1, 3).to(device),
                th.eye(3).unsqueeze(0).to(device),
                th.eye(2).unsqueeze(0).to(device),
                th.zeros(1, 2).to(device),
                output_filters=["render", "bary_img", "index_img", "mask"],
            )
            uwrapped_texs.append(out["render"][0])
            mask = out["mask"][0].float() * self.decoder.face_mask
            unwrapped_masks.append(mask[None])

        uwrapped_texs = th.stack(uwrapped_texs)
        unwrapped_masks = th.stack(unwrapped_masks)
        return {
            'uw_tex': uwrapped_texs,
            # 'uw_mask': unwrapped_masks,
            'img': imgs,
            # 'uv_img': uv_imgs,
            # 'mask': masks,
        }

    def cache_conditioning(self, batch, idxs):
        # raise NotImplementedError
        M = self.cond_codes.shape[0]
        B = len(idxs)
        N = len(self.cameras)
        ident = [ide for i, ide in enumerate(batch['index']['ident']) if i in idxs]

        if 'mean_' in self.cond_type:
            # operate on BMN data
            assert 'front_' not in self.cond_type
            hmc_Rt = batch['hmc_Rt'][idxs][:, None].expand(-1, M, -1, -1, -1).flatten(0, 1)
            hmc_K = batch['hmc_K'][idxs][:, None].expand(-1, M, -1, -1, -1).flatten(0, 1)
            rtvec = self.mean_rtvec[None].expand(B * M, -1)
            hmc_K = hmc_K.contiguous()
            hmc_Rt = hmc_Rt.contiguous()
            camera_params = gather_camera_params(
                rtvec,
                hmc_Rt,
                hmc_K,
                (self.tex_inp_size, self.tex_inp_size),
                isMouthView=self.isMouthView,
            )
            sc = 1
            camera_params['focal'] = camera_params['focal'] * sc

            cond_codes = self.cond_codes[None, :, None].expand(B, -1, N, -1).flatten(0, 2)
            if self.cond_perid_centering:
                mean_expr = [self.perid_stats[k[:6]]['mean'] for k in ident]
                mean_expr = th.stack(mean_expr).to(cond_codes.device)  # BD
                mean_expr = mean_expr[:, None, None].expand(-1, M, N, -1).flatten(0, 2)
                cond_codes = cond_codes + mean_expr
            ret = self.unwrap_texture(
                [ide for ide in ident for _ in range(M) for _ in range(N)],
                cond_codes,
                camera_params,
            )
            # (BNM)3HW -> BNM3HW
            ret = {k: v.unflatten(0, (B, M, N)).transpose(1, 2).detach() for k, v in ret.items()}

        elif 'front_' in self.cond_type:
            raise AttributeError
            assert 'mean_' not in self.cond_type
            camera_params = self.front_view_generator(B * M)
            cond_codes = self.cond_codes[None].expand(B, -1, -1).flatten(0, 1)
            if self.cond_perid_centering:
                mean_expr = [self.perid_stats[k[:6]]['mean'] for k in ident]
                mean_expr = th.stack(mean_expr).to(cond_codes.device)  # BD
                mean_expr = mean_expr[:, None].expand(-1, M, -1).flatten(0, 1)
                cond_codes = cond_codes + mean_expr
            ret = self.unwrap_texture(
                [ide for ide in ident for _ in range(M)], cond_codes, camera_params
            )  # (BM)3HW

            # (BM)3HW -> BNM3HW
            ret = {
                k: v.unflatten(0, (B, M))[:, None].expand(-1, N, -1, -1, -1, -1).detach()
                for k, v in ret.items()
            }

        else:
            raise NotImplementedError

        # resize texture before caching
        for k in ['uw_tex', 'uw_mask']:
            if k not in ret:
                continue
            ret[k] = resize2d(ret[k], (self.tex_inp_size, self.tex_inp_size))

        # cache each identity
        for i, ide in enumerate(ident):
            self.cond_info[ide] = {k: v[i] for k, v in ret.items()}

    def get_gt_slop_cond(self, batch):
        raise AttributeError
        N = len(self.cameras)
        ident = batch['index']['ident']
        B = len(ident)
        M = self.cond_codes.shape[0] + int(self.gt_expr)

        rosetta_state: State = batch['rosetta_correspondences']['state']
        hmc_Rt = batch['hmc_Rt'][:, None].expand(-1, M, -1, -1, -1).flatten(0, 1)
        hmc_K = batch['hmc_K'][:, None].expand(-1, M, -1, -1, -1).flatten(0, 1)
        rtvec = rosetta_state.rtvec[:, None].expand(-1, M, -1).flatten(0, 1)
        camera_params = gather_camera_params(
            rtvec.contiguous(),
            hmc_Rt.contiguous(),
            hmc_K.contiguous(),
            (self.tex_inp_size, self.tex_inp_size),
            isMouthView=self.isMouthView,
        )
        sc = 1
        camera_params['focal'] = camera_params['focal'] * sc

        expr = self.cond_codes[None, :, None].expand(B, -1, N, -1)  # B(M-1)ND
        if self.gt_expr:
            expr = th.cat([expr, rosetta_state.expr[:, None, None].expand(-1, -1, N, -1)], 1)
        if self.cond_perid_centering:
            mean_expr = [self.perid_stats[k[:6]]['mean'] for k in ident]
            mean_expr = th.stack(mean_expr).to(expr.device)  # BD
            mean_expr = expr + mean_expr[:, None, None]
        expr = expr.flatten(0, 2)

        ret = self.unwrap_texture(
            [ide for ide in ident for _ in range(M) for _ in range(N)],
            expr,
            camera_params,
        )
        ret = {k: v.unflatten(0, (B, M, N)).transpose(1, 2).detach() for k, v in ret.items()}
        # resize texture before caching
        for k in ['uw_tex', 'uw_mask']:
            if k not in ret:
                continue
            ret[k] = resize2d(ret[k], (self.tex_inp_size, self.tex_inp_size))

        return ret

    def get_gt_expr_cond(self, batch):
        raise AttributeError
        N = len(self.cameras)
        ident = batch['index']['ident']
        B = len(ident)

        if 'mean_' in self.cond_type:
            assert 'front_' not in self.cond_type
            rtvec = self.mean_rtvec[None].expand(B, -1)
            camera_params = gather_camera_params(
                rtvec,
                batch['hmc_Rt'],
                batch['hmc_K'],
                (self.tex_inp_size, self.tex_inp_size),
                isMouthView=self.isMouthView,
            )
            sc = 1
            camera_params['focal'] = camera_params['focal'] * sc
        elif 'front_' in self.cond_type:
            assert 'mean_' not in self.cond_type
            camera_params = self.front_view_generator(B * N)
        else:
            raise NotImplementedError

        rosetta_state: State = batch['rosetta_correspondences']['state']
        ret = self.unwrap_texture(
            [ide for ide in ident for _ in range(N)],
            rosetta_state.expr[:, None].expand(-1, N, -1).flatten(0, 1),
            camera_params,
        )
        # (BN)3HW -> BN3HW
        ret = {k: v.unflatten(0, (B, N)).detach() for k, v in ret.items()}

        # resize texture
        for k in ['uw_tex', 'uw_mask']:
            if k not in ret:
                continue
            ret[k] = resize2d(ret[k], (self.tex_inp_size, self.tex_inp_size))

        return ret

    def get_sdm_cond(self, batch):
        M = self.cond_codes.shape[0]
        N = len(self.cameras)
        ident = batch['index']['ident']
        B = len(ident)
        m = M + int(self.sdm_expr)

        hmc_Rt = batch['hmc_Rt'][:, None].expand(-1, m, -1, -1, -1).flatten(0, 1)
        hmc_K = batch['hmc_K'][:, None].expand(-1, m, -1, -1, -1).flatten(0, 1)
        rtvec = batch['sdm_rtvec'][:, None].expand(-1, m, -1).flatten(0, 1)
        camera_params = gather_camera_params(
            rtvec.contiguous(),
            hmc_Rt.contiguous(),
            hmc_K.contiguous(),
            tuple(self.render_size),
            isMouthView=self.isMouthView,
        )

        cond_codes = self.cond_codes[None, :, None].expand(B, -1, N, -1)  # BMND
        if self.cond_perid_centering:
            mean_expr = [self.perid_stats[k[:6]]['mean'] for k in ident]
            mean_expr = th.stack(mean_expr).to(cond_codes.device)  # BD
            cond_codes = cond_codes + mean_expr[:, None, None, :]  # BMND

        expr = cond_codes
        if self.sdm_expr:
            sdm_expr = batch['sdm_expr'][:, None, None].expand(-1, -1, N, -1)  # B1ND
            expr = th.cat([expr, sdm_expr], 1)
        expr = expr.flatten(0, 2)  # BmND

        ret = self.unwrap_texture(
            [ide for ide in ident for _ in range(m) for _ in range(N)], expr, camera_params
        )
        ret = {k: v.unflatten(0, (B, m, N)).transpose(1, 2).detach() for k, v in ret.items()}

        # resize texture
        for k in ['uw_tex', 'uw_mask']:
            if k not in ret:
                continue
            ret[k] = resize2d(ret[k], (self.tex_inp_size, self.tex_inp_size))
        return ret

    def get_conditioning(self, batch):
        if 'gtslop' in self.cond_type:
            id_cond = self.get_gt_slop_cond(batch)
            batch['id_cond'] = id_cond
            return batch

        if 'sdm' in self.cond_type:
            id_cond = self.get_sdm_cond(batch)
            batch['id_cond'] = id_cond
            return batch

        ident = batch['index']['ident']
        idx_not_avail = [i for i, ide in enumerate(ident) if ide not in self.cond_info]
        if len(idx_not_avail) > 0:
            self.eval()
            with th.no_grad():
                for idx in idx_not_avail:
                    self.cache_conditioning(batch, [idx])

        if self.gt_expr:
            gt_expr_cond = self.get_gt_expr_cond(batch)

        assert 'id_cond' not in batch
        batch['id_cond'] = {}
        for k in ['img', 'mask', 'uw_tex', 'uw_mask', 'uv_img']:
            if k not in self.cond_info[ident[0]]:
                continue
            batch['id_cond'][k] = th.stack([self.cond_info[ide][k] for ide in ident])  # BNM3HW

            if self.gt_expr:
                batch['id_cond'][k] = th.cat([batch['id_cond'][k], gt_expr_cond[k][:, :, None]], 2)

        return batch
