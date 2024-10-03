from typing import Sequence
import torch as th
import time
import torch.nn.functional as thf
from drtk.renderlayer.projection import project_points
from drtk.renderlayer.renderlayer import RenderLayer as RenderLayer

# import IPython  # pylint: disable=unused-import

from utils import (
    put_text_on_img,
    gather_camera_params,
    # frontal_view_camera_params,
    get_error,
    load_rosetta_stats,
    # gradient_postprocess,
    get_angle_between_rot,
    slice_batch,
    get_num_iters,
    # undistort_hmc_cam_img_batch,
    resize2d,
)
from sdm_utils import (
    State,
    preprocess_batch,
    splitbatch_decoder_forward,
    paper_diag_input,
    paper_diag_state,
    color_correct_2,
)
from keypoint import KeypointHandler


class SDM_Module(base_classes.Module):
    def __init__(self, decoder, **kwargs) -> None:
        super().__init__()
        self.decoder = decoder
        print(kwargs)
        self.cameras = kwargs.get('cameras')
        self.isMouthView = [('mouth' in c) for c in self.cameras]

        # ------------------------------------
        # Keypoint
        # ------------------------------------
        self.kp_handler = KeypointHandler(cameras=self.cameras, **kwargs.get('keypoint_kwargs'))

        # ------------------------------------
        # Configuration
        # ------------------------------------
        self.expr_dim = 256
        self.rotdim = kwargs.get("rotdim", 6)
        self.render_size = kwargs.get("render_size")
        self.rosetta_keypoints = kwargs.get("rosetta_keypoints")
        self.predict_slope = kwargs.get("predict_slope")
        self.predict_expr = kwargs.get("predict_expression")
        self.num_iters = kwargs.get("num_iters")
        self.num_val_iters = kwargs.get("num_val_iters", self.num_iters)
        self.loss_cfg = kwargs.get("loss_cfg")
        self.iter_loss_weight_type = kwargs.get("iter_loss_weight_type")
        assert self.iter_loss_weight_type in ["none", "prev_iter_loss"]
        self.input_current_state = kwargs.get("input_current_state")
        self.input_guiding_feature = kwargs.get("input_guiding_feature")
        self.input_residual = kwargs.get("input_residual")
        self.guiding_feature_type = kwargs.get("guiding_feature_type")
        self.num_iters_schedule = kwargs.get("num_iters_schedule")
        self.expr_normalize_type = kwargs.get("expr_normalize_type")
        self.init_noise = kwargs.get("init_noise")
        assert self.expr_normalize_type in ['global_meanstd', 'perid_mean', 'global_covar']
        self.viz_keypoints = True
        assert self.init_noise in [True, False]

        # ------------------------------------
        # Model
        # ------------------------------------
        self.num_pred_params = self.predict_expr * self.expr_dim + self.predict_slope * (
            3 + self.rotdim
        )
        assert self.num_pred_params > 0
        self.set_model(**kwargs)
        if self.model:
            print(self.model)
            print("number of parameters", sum(p.numel() for p in self.model.parameters()))

        # ------------------------------------
        # Stats
        # ------------------------------------
        stats = load_rosetta_stats()
        rotkey = 'avatar2hmc_rot6d' if self.rotdim == 6 else 'avatar2hmc_rvec'
        mean_rtvec = th.cat([stats[rotkey]['mean'], stats['avatar2hmc_tvec']['mean']])
        std_rtvec = th.cat([stats[rotkey]['std'], stats['avatar2hmc_tvec']['std']])
        self.register_buffer("mean_rtvec", mean_rtvec, persistent=False)
        self.register_buffer("std_rtvec", std_rtvec, persistent=False)
        self.register_buffer("mean_expr", stats['expression']['mean'], persistent=False)
        self.register_buffer("std_expr", stats['expression']['std'], persistent=False)
        # self.register_buffer("WT_expr", stats['expression']['WT'], persistent=False)
        # self.register_buffer("WTinv_expr", stats['expression']['WT'].inverse(), persistent=False)
        if self.expr_normalize_type == 'perid_mean':
            self.perid_stats = stats['perid_stats']
            # for k, v in perid_stats.items():
            #     self.register_buffer(f"{k}_expr_mean", v['mean'], persistent=False)
            #     self.register_buffer(f"{k}_expr_std", v['std'], persistent=False)
        else:
            raise AttributeError

        # ------------------------------------
        # Checkpoint
        # ------------------------------------
        if kwargs.get('ckpt_path') is not None:
            print("Loading ", kwargs['ckpt_path'])
            dct = th.load(kwargs['ckpt_path'], map_location='cpu')
            dct = {k.replace('model.', ''): v for k, v in dct.items()}
            if 'resnet.stem.conv1.weight' in dct:
                ckpt_nc = dct['resnet.stem.conv1.weight'].shape[1]
                model_nc = self.model.resnet.stem.conv1.weight.shape[1]
                if ckpt_nc < model_nc:
                    tmp = [
                        dct['resnet.stem.conv1.weight'],
                        self.model.resnet.stem.conv1.weight[:, ckpt_nc:] * 0.01,
                    ]
                    dct['resnet.stem.conv1.weight'] = th.cat(tmp, 1)
                    print("Only some channels of the cnn stem were initialized.")
                elif ckpt_nc > model_nc:
                    dct['resnet.stem.conv1.weight'] = dct['resnet.stem.conv1.weight'][:, :model_nc]
                    print("Some ckpt cnn stem channels were discarded.")
            self.model.load_state_dict(dct)

        # ------------------------------------
        # for generating uv map
        # ------------------------------------
        topology_file = kwargs.get("tex_topology_file")
        topology = typed.load(get_repo_asset_path(topology_file))
        self.renderer = RenderLayer(
            400,
            400,
            topology["vt"],
            topology["vi"],
            topology["vti"],
            boundary_aware=False,
            flip_uvs=False,
        )
        u, v = th.meshgrid(th.linspace(-1, 1, 1024), th.linspace(-1, 1, 1024), indexing='xy')
        self.register_buffer("uvcoord", th.stack([u, v, th.ones_like(u)])[None], persistent=False)

        # ------------------------------------
        # Misc
        # ------------------------------------
        self.register_buffer(
            "color_map", th.from_numpy(get_color_map("COLORMAP_JET")).byte(), persistent=False
        )
        self.front_view_generator = FixedFrontalViewGenerator()
        self.track_time = False

    def set_model(self, **kwargs):
        raise NotImplementedError

    def get_optim_params(self, base_lr: float):
        return [{"name": "sdm", "lr": base_lr, "params": self.model.parameters()}]

    def fk(
        self, batch, rtvec, expr, use_distort=False, render=False, onepass=False, get_uv_img=False
    ) -> State:
        B, N = batch["hmc_cam_img"].shape[:2]

        if use_distort:
            assert not th.is_grad_enabled()
            dist_params = {k: batch[k] for k in ['hmc_distmodel', 'hmc_distcoeffs']}
            raise AttributeError("Don't use distortion in style transfer")
        else:
            dist_params = {}

        img, mask, uv_img = None, None, None
        front_img, front_mask = None, None
        if render:
            ident = [ide for ide in batch['index']['ident'] for _ in range(N)]
            camera_params = gather_camera_params(
                rtvec,
                batch['hmc_Rt'],
                batch['hmc_K'],
                self.render_size,
                **dist_params,
                isMouthView=self.isMouthView,
            )

            decoder_out = splitbatch_decoder_forward(
                self.decoder,
                ident,
                expr[:, None].expand(-1, N, -1).flatten(0, 1),
                camera_params,
                onepass=onepass,
            )
            geo_pred = decoder_out["geo_pred"]
            verts_a = decoder_out["geo_pred"].view(B, N, -1, 3).mean(1)
            img = decoder_out['img_pred'].view(B, N, 3, *self.render_size)
            mask = decoder_out['mask'].view(B, N, 1, *self.render_size)

            if get_uv_img:
                self.renderer.resize(self.render_size[0], self.render_size[1])
                camera_params.pop('render_h')
                camera_params.pop('render_w')
                camera_params.pop('isMouthView')
                render_out = self.renderer(
                    geo_pred,
                    self.uvcoord.expand(B * N, -1, -1, -1),
                    output_filters=["bary_img", "index_img", "render"],
                    **camera_params,
                )
                uv_img = render_out['render']  # (BN)3HW
                uv_img = th.stack([uv_img[:, 0], -uv_img[:, 1]], 1)  # (BN)2HW
                uv_img = uv_img.unflatten(0, (B, N)) * mask
                mask2 = mask.expand(-1, -1, 2, -1, -1)
                uv_img = th.where(mask2 > 0.5, uv_img, uv_img.sign())
        else:
            if not self.predict_expr and ('state' in batch['rosetta_correspondences']):
                verts_a = batch['rosetta_correspondences']['state'].verts_a
            else:
                verts_a = self.decoder.forward_geo(batch, expr)

        kp3d_a = self.kp_handler.get_keypoints(verts_a)
        kp3d_a_repeated = kp3d_a[:, None].expand(-1, N, -1, -1).flatten(0, 1)
        camera_params = gather_camera_params(
            rtvec, batch['hmc_Rt'], batch['hmc_K'], None, **dist_params
        )
        kp2d, kp3d = project_points(kp3d_a_repeated, **camera_params)
        # verts = (
        #     camera_params['camrot'][:, None] @
        #     (verts_a[:, None].expand(-1, N, -1, -1).flatten(0, 1) - camera_params['campos'][:, None])[..., None]
        # )[..., 0]

        kp2d_vis = batch['hmc_keypoints'][..., 2:].view(B * N, -1, 1)
        kp2d = th.cat((kp2d[..., :-1], kp2d_vis), dim=-1)

        kp2d = kp2d.reshape(B, N, -1, 3)
        kp3d = kp3d.reshape(B, N, -1, 3)
        if self.render_size[0] != 400 or self.render_size[1] != 400:
            sc = [self.render_size[0] / 400, self.render_size[1] / 400, 1]
            sc = th.tensor(sc).float().to(kp2d.device)
            kp2d = kp2d * sc[None, None, None]

        # verts = verts.reshape(B, N, -1, 3)
        return State(
            rtvec=rtvec,
            expr=expr,
            verts_a=verts_a,
            # kp3d_a=kp3d_a,
            # verts=verts,
            kp3d=kp3d,
            kp2d=kp2d,
            img=img,
            mask=mask,
            use_distort=use_distort,
            uv_img=uv_img,
            front_img=front_img,
            front_mask=front_mask,
        )

    def add_front_image(self, batch, states: Sequence[State]):
        states = [s for s in states if s.front_img is None]
        if len(states) == 0:
            return
        B = states[0].rtvec.shape[0]
        # fcam = frontal_view_camera_params(B * len(states), states[0].rtvec.device, scale=0.25)
        # fcam.pop('distortion_mode')
        # fcam.pop('distortion_coeff')
        fcam = self.front_view_generator.forward(B * len(states))
        decoder_out, _ = self.decoder.forward(
            data={'index': {'ident': batch['index']['ident'] * len(states)}},
            inputs={
                'expression': th.cat([s.expr for s in states], 0),
                'camera_params': fcam,
            },
            compute_losses=False,
        )
        imgs = decoder_out['img_pred'].split([B] * len(states), 0)
        masks = decoder_out['mask'].split([B] * len(states), 0)
        for im, m, s in zip(imgs, masks, states):
            s.front_img = im
            s.front_mask = m

    def get_perid_expr_mean(self, batch):
        if 'perid_expr_mean' not in batch:
            mean = [self.perid_stats[k[:6]]['mean'] for k in batch['index']['ident']]
            batch['perid_expr_mean'] = th.stack(mean).to(batch['hmc_cam_img'].device)
        return batch['perid_expr_mean']

    def get_init_state(self, batch) -> State:
        B = batch["hmc_cam_img"].shape[0]
        rosetta_gt = batch['rosetta_correspondences']

        if "rtvec_init" in batch:
            assert self.predict_expr and self.predict_slope
            rtvec = batch["rtvec_init"]
            expr = batch["expr_init"]
            return self.fk(batch, rtvec, expr)

        if self.predict_slope:
            rtvec = self.mean_rtvec[None].expand(B, -1)
        else:
            rtvec = rosetta_gt['state'].rtvec

        if self.predict_expr:
            expr = th.zeros(B, self.expr_dim, device=rtvec.device)
            if self.expr_normalize_type == 'perid_mean':
                expr = expr + self.get_perid_expr_mean(batch)
        else:
            expr = rosetta_gt['expression']
        return self.fk(batch, rtvec, expr)

    def guiding_feature(self, batch, state: State):
        raise NotImplementedError

    def data_feature(self, batch, state: State):
        raise NotImplementedError

    def state_feature(self, batch, state: State):
        sfeat = []
        if self.predict_slope:
            rtvec = (state.rtvec - self.mean_rtvec) / self.std_rtvec
            sfeat.append(rtvec)
        if self.predict_expr:
            if self.expr_normalize_type == 'global_meanstd':
                expr = (state.expr - self.mean_expr) / self.std_expr
            elif self.expr_normalize_type == 'global_covar':
                expr = (state.expr - self.mean_expr) @ self.WT_expr.T
            elif self.expr_normalize_type == 'perid_mean':
                expr = state.expr - self.get_perid_expr_mean(batch)
            sfeat.append(expr)
        return th.cat(sfeat, -1) * int(self.input_current_state)

    def update_state(self, batch, state, out):
        if self.predict_slope:
            delta_rtvec = out[..., : self.rotdim + 3]
            delta_rtvec = delta_rtvec * self.std_rtvec
            rtvec = delta_rtvec + state.rtvec
        else:
            rtvec = state.rtvec

        if self.predict_expr:
            delta_expr = out[..., -self.expr_dim :]
            if self.expr_normalize_type == 'global_meanstd':
                delta_expr = delta_expr * self.std_expr
            elif self.expr_normalize_type == 'global_covar':
                delta_expr = delta_expr @ self.WTinv_expr.T
            elif self.expr_normalize_type == 'perid_mean':
                pass
            expr = delta_expr + state.expr
        else:
            expr = state.expr

        return self.fk(batch, rtvec, expr)

    def preprocess_batch_and_rosetta_gt(self, batch):
        if 'state' in batch['rosetta_correspondences']:
            return batch
        # clear confidence of keypoints that are not visible for a particular view
        # batch['hmc_keypoints'] = self.kp_handler.select_subset(batch['hmc_keypoints'])
        # batch['hmc_keypoints'] = self.kp_handler.scatter_subset(batch['hmc_keypoints'])
        batch = preprocess_batch(batch)

        # ------------------------------------
        # prepare rosetta_state for target
        # ------------------------------------
        rosetta_gt = batch['rosetta_correspondences']
        rotkey = 'avatar2hmc_rot6d' if self.rotdim == 6 else 'avatar2hmc_rvec'
        rosetta_rtvec = th.cat([rosetta_gt[rotkey], rosetta_gt['avatar2hmc_tvec']], -1)
        rosetta_state = self.fk(batch, rosetta_rtvec, rosetta_gt['expression'], get_uv_img=True)
        batch['rosetta_correspondences']['state'] = rosetta_state

        # ------------------------------------
        # resize hmc image and keypoints
        # ------------------------------------
        batch['hmc_cam_img'] = resize2d(batch['hmc_cam_img'].float(), self.render_size)

        sc = [self.render_size[0] / 400, self.render_size[1] / 400, 1]
        sc = th.tensor(sc).float().to(batch['hmc_keypoints'].device)
        batch['hmc_keypoints'] = batch['hmc_keypoints'] * sc[None, None, None]

        # ------------------------------------
        # use clean keypoints if specified
        # ------------------------------------
        if self.rosetta_keypoints:
            # rsud = self.fk(batch, rosetta_rtvec, rosetta_gt['expression'], render=False)
            # rsud.kp2d[..., -1] = 1
            # batch['hmc_keypoints'] = rsud.kp2d
            batch['hmc_keypoints'] = rosetta_state.kp2d

        return batch

    def forward(
        self, batch, compute_losses=False, num_iters=None
    ):  # pylint: disable=arguments-renamed
        batch = self.preprocess_batch_and_rosetta_gt(batch)
        if "results_m2" in batch:
            st_img_check = resize2d(batch["results_m2"]["pred_rgb"].float(), self.render_size)
            batch["st_img_check"] = st_img_check

            rtvec = batch["results_m2"]["rtvec"][:, -1]
            expr = batch["results_m2"]["expr"][:, -1]
            if self.training and self.init_noise:
                rtvec = rtvec + th.randn_like(rtvec) * self.std_rtvec * 0.15
                expr = expr + th.randn_like(expr) * self.std_expr * 0.3
            batch["rtvec_init"] = rtvec
            batch["expr_init"] = expr

        start_time = time.time()
        state = self.get_init_state(batch)
        state_iters = [state]

        if num_iters is None:
            num_iters = (
                get_num_iters(batch['iteration'], self.num_iters, self.num_iters_schedule)
                if self.training
                else self.num_val_iters
            )
        for iter_idx in range(num_iters):  # pylint: disable=unused-variable
            state = state.detach()
            dfeat = self.data_feature(batch, state)
            sfeat = self.state_feature(batch, state)
            gfeat = self.guiding_feature(batch, state)
            out = self.model.forward([dfeat, sfeat, gfeat], iter_idx)
            state = self.update_state(batch, state, out)
            state_iters.append(state)

        taken_time = time.time() - start_time
        if self.track_time:
            print("SDM forward took", taken_time)

        if compute_losses:
            losses = self.compute_losses(batch, state_iters)
            return state_iters, losses
        else:
            return state_iters

    def compute_losses(self, batch, state_iters: Sequence[State], get_metrics=True):
        B, N = batch['hmc_cam_img'].shape[:2]

        # pylint: disable=unused-variable
        det_kp2d = batch['hmc_keypoints'].view(B, N, -1, 3)
        rosetta_state: State = batch['rosetta_correspondences']['state']

        lweight = self.loss_cfg.weight
        ltype = self.loss_cfg.type
        losses = {k: [] for k, v in lweight.items() if v > 1.0e-9}

        if 'fimg' in losses:
            self.add_front_image(batch, state_iters + [rosetta_state])

        for sidx, state in enumerate(state_iters):
            assert not state.use_distort and not rosetta_state.use_distort

            if 'kp3d' in losses:
                kp3d_loss = get_error(
                    state.kp3d / 1000,
                    rosetta_state.kp3d / 1000,
                    # det_kp2d[..., 2:],
                    None,
                    ltype['kp3d'],
                    [1, 2, 3],
                )
                losses['kp3d'].append(kp3d_loss)

            if 'kp3dcen' in losses:
                a = state.kp3d / 1000
                b = rosetta_state.kp3d / 1000
                kp3dcen_loss = get_error(
                    a - a.mean(2, keepdim=True),
                    b - b.mean(2, keepdim=True),
                    det_kp2d[..., 2:],
                    ltype['kp3dcen'],
                    [1, 2, 3],
                )
                kp3dcen_loss += get_error(
                    a.mean(2), b.mean(2), None, ltype['kp3dcen'], sum_dim=[1, 2]
                )
                losses['kp3dcen'].append(kp3dcen_loss)

            if 'verts' in losses:
                verts_loss = get_error(
                    state.verts / 1000,
                    rosetta_state.verts / 1000,
                    loss_type=ltype['verts'],
                    sum_dim=[1, 2, 3],
                )
                losses['verts'].append(verts_loss)

            if 'rvec' in losses:
                assert self.predict_slope
                rvec_loss = get_error(
                    state.rtvec[..., : self.rotdim],
                    rosetta_state.rtvec[..., : self.rotdim],
                    loss_type=ltype['rvec'],
                    sum_dim=[1],
                )
                losses['rvec'].append(rvec_loss)

            if 'tvec' in losses:
                assert self.predict_slope
                tvec_loss = get_error(
                    state.rtvec[..., self.rotdim :],
                    rosetta_state.rtvec[..., self.rotdim :],
                    loss_type=ltype['tvec'],
                    sum_dim=[1],
                )
                losses['tvec'].append(tvec_loss)

            if 'expr' in losses:
                assert self.predict_expr
                expr_loss = get_error(
                    state.expr, rosetta_state.expr, loss_type=ltype['expr'], sum_dim=[1]
                )
                losses['expr'].append(expr_loss)

            if 'img' in losses:
                assert ltype['img'] == 'l1'
                diff = thf.smooth_l1_loss(state.img, rosetta_state.img, reduction="none")
                mask = state.mask * rosetta_state.mask
                img_loss = (diff * mask).mean((1, 2, 3, 4)) / mask.mean((1, 2, 3, 4))
                losses['img'].append(img_loss)

            if 'fimg' in losses:
                assert ltype['fimg'] == 'l1'
                diff = thf.smooth_l1_loss(
                    state.front_img, rosetta_state.front_img, reduction="none"
                )
                mask = state.front_mask * rosetta_state.front_mask
                fimg_loss = (diff * mask).mean((1, 2, 3)) / mask.mean((1, 2, 3))
                losses['fimg'].append(fimg_loss)

        losses = {k: th.stack(v, 1) for k, v in losses.items() if len(v) > 0}  # B x (num_iters+1)

        total_loss = 0  # B x (num_iters+1)
        for k, v in losses.items():
            total_loss += lweight[k] * v

        if len(losses) > 0:
            iter_lw = self.get_iter_loss_weights(total_loss)  # B x num_iters
            total_loss = (total_loss[:, 1:] * iter_lw).mean(1)  # B

        losses = {f'{k}_{ltype[k]}': v[:, 1:].mean(1) for k, v in losses.items()}
        losses['total'] = total_loss
        if get_metrics:
            with th.no_grad():
                metrics = self.compute_metrics(batch, state_iters)
            losses.update(metrics)
        return losses

    def get_iter_loss_weights(self, loss):
        # loss: B x (num_iters+1)
        if self.iter_loss_weight_type == "none":
            return th.ones_like(loss[..., 1:])
        elif self.iter_loss_weight_type == "prev_iter_loss":
            fact = loss[:, 1:] / (loss[:, :-1] + 1.0e-8)  # B x num_iters
            fact = fact.detach()
            fact = fact.clamp(10, 0.1)
            fact = fact.shape[1] * fact / fact.sum(1, keepdim=True)
            return fact
        else:
            raise NotImplementedError

    def compute_metrics(self, batch, state_iters: Sequence[State]):
        state = state_iters[-1]
        rosetta_state: State = batch['rosetta_correspondences']['state']
        det_kp2d = self.kp_handler.select_subset(batch['hmc_keypoints'])
        state_kp2d = self.kp_handler.select_subset(state.kp2d)
        rosetta_kp2d = self.kp_handler.select_subset(rosetta_state.kp2d)

        det_kp2d_err = (state_kp2d - det_kp2d)[..., :2].square().sum(-1).sqrt().mean([1, 2])
        rosetta_kp2d_err = (state_kp2d - rosetta_kp2d)[..., :2].square().sum(-1).sqrt().mean([1, 2])
        kp3d_err = (state.kp3d - rosetta_state.kp3d).square().sum(-1).sqrt().mean([1, 2])
        rvec_err = get_angle_between_rot(
            state.rtvec[..., : self.rotdim], rosetta_state.rtvec[..., : self.rotdim]
        )
        tvec_err = (
            (state.rtvec[..., self.rotdim :] - rosetta_state.rtvec[..., self.rotdim :])
            .square()
            .sum(-1)
            .sqrt()
        )
        metrics = {
            'metric_det_kp2d': det_kp2d_err,
            'metric_rosetta_kp2d': rosetta_kp2d_err,
            'metric_kp3d': kp3d_err,
            'metric_rvec': rvec_err,
            'metric_tvec': tvec_err,
        }
        if state.img is not None and not self.training and not th.is_grad_enabled():
            self.add_front_image(batch, [state, rosetta_state])
            diff = thf.smooth_l1_loss(state.front_img, rosetta_state.front_img, reduction="none")
            mask = state.front_mask * rosetta_state.front_mask
            img_err = (diff * mask).mean((1, 2, 3)) / mask.mean((1, 2, 3))
            metrics['metric_front_img'] = img_err
        return metrics

    def compute_eval(self, batch, state_iters: Sequence[State]):
        """For Shaojie's evaluation script. Used by sdm_eval_m1/m3 stages."""
        image_L2 = []
        image_L1 = []
        geo_weighted_L1 = []
        rvec_deg = []
        tvec_mm = []
        rosetta_state: State = batch['rosetta_correspondences']['state']

        for state in state_iters[1:]:
            rvec_deg_i = get_angle_between_rot(
                state.rtvec[..., : self.rotdim], rosetta_state.rtvec[..., : self.rotdim]
            )
            rvec_deg.append(rvec_deg_i)
            tvec_mm_i = (
                (state.rtvec[..., self.rotdim :] - rosetta_state.rtvec[..., self.rotdim :])
                .square()
                .sum(-1)
                .sqrt()
            )
            tvec_mm.append(tvec_mm_i)

            self.add_front_image(batch, [state, rosetta_state])
            diff = thf.smooth_l1_loss(state.front_img, rosetta_state.front_img, reduction="none")
            mask = state.front_mask * rosetta_state.front_mask
            image_L1_i = (diff * mask).mean((1, 2, 3)) / mask.mean((1, 2, 3))
            image_L1.append(image_L1_i)

            diff = (state.front_img - rosetta_state.front_img).square()
            image_L2_i = (diff * mask).mean((1, 2, 3)) / mask.mean((1, 2, 3))
            image_L2.append(image_L2_i)

            geo_weighted_L1_i = thf.smooth_l1_loss(
                state.verts_a * self.decoder.face_weight_geo[:, None],
                rosetta_state.verts_a * self.decoder.face_weight_geo[:, None],
                reduction="none",
            ).mean((-1, -2))
            geo_weighted_L1.append(geo_weighted_L1_i)

        image_L1 = th.stack(image_L1, 1).detach().cpu().numpy().tolist()
        image_L2 = th.stack(image_L2, 1).detach().cpu().numpy().tolist()
        geo_weighted_L1 = th.stack(geo_weighted_L1, 1).detach().cpu().numpy().tolist()
        rvec_deg = th.stack(rvec_deg, 1).detach().cpu().numpy().tolist()
        tvec_mm = th.stack(tvec_mm, 1).detach().cpu().numpy().tolist()

        ret = [
            {
                'image_L1': image_L1[i],
                'image_L2': image_L2[i],
                'geo_weighted_L1': geo_weighted_L1[i],
                'rvec_deg': rvec_deg[i],
                'tvec_mm': tvec_mm[i],
            }
            for i in range(len(image_L1))
        ]
        return ret

    def visualize_state(
        self, batch, state: State, title=None, det_kp2d=None, rosetta_state: State = None
    ):
        fs = 1  # font scale
        B, N = batch['hmc_cam_img'].shape[:2]
        if state.img is None:
            state = self.fk(batch, state.rtvec, state.expr, render=True)

        # if self.aug_module is not None and rosetta_state is not None:
        #     old_state_img = state.img
        #     old_rosetta_state_img = rosetta_state.img
        #     state.img, rosetta_state.img = self.aug_module.augment_hmc(state.img, rosetta_state.img)

        # Visualize HMC views
        # BNCHW and then BHNWC -> BH(NW)C
        B, N, _, H, W = state.img.shape
        if self.viz_keypoints:
            state_kp_img = self.kp_handler.visualize_hmc_keypoints(state.kp2d, state.img, det_kp2d)
        else:
            state_kp_img = state.img.permute(0, 3, 1, 4, 2).flatten(2, 3)

        # Visualize front view
        self.add_front_image(batch, [state])
        fH = self.render_size[0]
        fW = int(fH * state.front_img.shape[-1] / state.front_img.shape[-2])
        state_fimg = state.front_img.permute(0, 2, 3, 1)  # BHWC

        # Concat front view with HMC view
        state_kp_img = th.cat(
            [state_kp_img, resize2d(state.front_img, (fH, fW)).permute(0, 2, 3, 1)],
            2,
        )

        if title is not None:
            state_kp_img = put_text_on_img(state_kp_img, title, fs)
            state_fimg = put_text_on_img(state_fimg.contiguous(), title, fs)

        # Error map with rosetta images
        if rosetta_state is not None:
            assert rosetta_state.use_distort == state.use_distort
            # Error map in HMC views
            mask = state.mask * rosetta_state.mask
            diff = (state.img - rosetta_state.img).mul(mask).abs().mean(-3) * 5
            diff = self.color_map[diff.clamp(0, 255).long()]
            diff = diff.permute(0, 2, 1, 3, 4).reshape(B, H, N * W, -1)  # BH(NW)C

            # Error map in front view
            self.add_front_image(batch, [rosetta_state])
            fmask = state.front_mask * rosetta_state.front_mask
            fdiff = (state.front_img - rosetta_state.front_img).mul(fmask).abs().mean(-3) * 5
            fdiff = self.color_map[fdiff.clamp(0, 255).long()]  # BHWC

            # Concat front error map with HMC view error map
            fdiff_small = fdiff.permute(0, 3, 1, 2).float()  # BCHW
            fdiff_small = (
                resize2d(fdiff_small, (fH, fW)).clamp(0, 255).long().permute(0, 2, 3, 1)
            )  # BHWC

            # Concat error maps with renderings
            state_kp_img = th.cat([state_kp_img, diff, fdiff_small], 2)  # BH(2NW)C
            state_fimg = th.cat([state_fimg, fdiff], 1)  # B(2H)WC
        # if rosetta_state is not None:
        #     state.img = old_state_img
        #     rosetta_state.img = old_rosetta_state_img
        return state_kp_img, state_fimg

    def get_diag_images(self, batch, state_iters, losses):  # pylint: disable=arguments-renamed
        idx = slice(0, min(8, batch['hmc_cam_img'].shape[0]))
        batch = slice_batch(batch, idx, ["hmc_distmodel", "camera"])
        state_iters = [s[idx] for s in state_iters]
        fs = 1  # font scale

        if 'state' not in batch['rosetta_correspondences']:
            batch = self.preprocess_batch_and_rosetta_gt(batch)

        _, N = batch['hmc_cam_img'].shape[:2]
        rosetta_state: State = batch['rosetta_correspondences']['state']
        if rosetta_state.img is None:
            rosetta_state = self.fk(batch, rosetta_state.rtvec, rosetta_state.expr, render=True)
        det_kp2d = batch['hmc_keypoints']

        state_diags = []
        state_fdiags = []
        for i, state in enumerate(state_iters):
            if i not in [0, 1, 2, 3, 4, len(state_iters) - 1]:
                continue
            state_diag, state_fdiag = self.visualize_state(
                batch, state, f'iter {i}', det_kp2d, rosetta_state
            )
            state_diags.append(state_diag)
            state_fdiags.append(state_fdiag)

        r_diag, r_fdiag = self.visualize_state(batch, rosetta_state, 'rosetta corr', det_kp2d)

        if self.viz_keypoints:
            det_kp2d_img = self.kp_handler.visualize_hmc_keypoints(det_kp2d, batch['hmc_cam_img'])
            det_kp2d_img = put_text_on_img(det_kp2d_img, 'detection', fs)  # BH(NW)C
        else:
            det_kp2d_img = batch['hmc_cam_img'].expand(-1, -1, 3, -1, -1)
            if 'st_img_check' in batch:
                bla = resize2d(batch['st_img_check'], self.render_size)
                det_kp2d_img = th.cat([det_kp2d_img, bla], 3)
            det_kp2d_img = det_kp2d_img.permute(0, 3, 1, 4, 2).flatten(2, 3)
            det_kp2d_img = put_text_on_img(det_kp2d_img, 'hmc inp', fs)  # BH(NW)C

        state_diags = pad_and_cat(state_diags + [r_diag, det_kp2d_img], 1)  # B(XH)(NW)C

        # BH(NW)C -> BHNWC -> BNHWC -> BV2HWC -> BVH2WC -> B(VH)(2W)C
        det_kp2d_img_grid = (
            det_kp2d_img.unflatten(2, (N, -1))
            .transpose(1, 2)
            .unflatten(1, (-1, 2))
            .transpose(2, 3)
            .flatten(3, 4)
            .flatten(1, 2)
        )
        state_fdiags = pad_and_cat(state_fdiags + [r_fdiag, det_kp2d_img_grid], 2)  # B(2H)(XW)C

        return {
            'hmc_front': state_diags,
            'front': state_fdiags,
        }

    def visualize_state_paper(
        self, batch, state: State, title=None, det_kp2d=None, rosetta_state: State = None
    ):
        """Tried to do white background for the paper."""
        # fs = 1  # font scale
        if state.img is None:
            state = self.fk(batch, state.rtvec, state.expr, render=True)

        # Visualize HMC views
        assert not self.viz_keypoints
        # state.img BN3HW
        state_kp_img = th.cat(
            [
                th.cat([state.img[:, 0], state.img[:, 1]], -1),
                th.cat([state.img[:, 2], state.img[:, 3]], -1),
            ],
            -2,
        )  # B3(2H)(2W)
        state_kp_img = state_kp_img.permute(0, 2, 3, 1)  # B(2H)(2W)3

        # Visualize front view
        self.add_front_image(batch, [state])
        fH = self.render_size[0] * 2
        fW = int(fH * state.front_img.shape[-1] / state.front_img.shape[-2])
        state_fimg = state.front_img.permute(0, 2, 3, 1)  # BHWC

        # Concat front view with HMC view
        state_kp_img = th.cat(
            [state_kp_img, resize2d(state.front_img, (fH, fW)).permute(0, 2, 3, 1)],
            2,
        )
        state_kp_img = color_correct_2(state_kp_img, -1)
        # state_kp_img = linear2color_corr(state_kp_img / 255, -1).clamp(0, 1) * 255

        if title is not None:
            # state_kp_img = put_text_on_img(state_kp_img, title, fs)
            # state_fimg = put_text_on_img(state_fimg.contiguous(), title, fs)
            pass

        # Error map with rosetta images
        if rosetta_state is not None:
            assert rosetta_state.use_distort == state.use_distort
            # Error map in HMC views
            mask = state.mask * rosetta_state.mask
            diff = (state.img - rosetta_state.img).mul(mask).abs().mean(-3) * 5
            diff = self.color_map[diff.clamp(0, 255).long()]  # BNHWC
            diff = th.cat(
                [
                    th.cat([diff[:, 0], diff[:, 1]], -2),
                    th.cat([diff[:, 2], diff[:, 3]], -2),
                ],
                -3,
            )  # B(2H)(2W)3

            # Error map in front view
            self.add_front_image(batch, [rosetta_state])
            fmask = state.front_mask * rosetta_state.front_mask
            fdiff = (state.front_img - rosetta_state.front_img).mul(fmask).abs().mean(-3) * 5
            fdiff = self.color_map[fdiff.clamp(0, 255).long()]  # BHWC

            # Concat front error map with HMC view error map
            fdiff_small = fdiff.permute(0, 3, 1, 2).float()  # BCHW
            fdiff_small = (
                resize2d(fdiff_small, (fH, fW)).clamp(0, 255).long().permute(0, 2, 3, 1)
            )  # BHWC

            # Concat error maps with renderings
            state_fimg = th.cat([state_fimg, fdiff], 1)  # B(2H)WC
            state_kp_img = th.cat([state_kp_img, th.cat([diff, fdiff_small], 2)], 1)  # BH(2NW)C
        # if rosetta_state is not None:
        #     state.img = old_state_img
        #     rosetta_state.img = old_rosetta_state_img
        return state_kp_img, state_fimg

    def get_diag_images_paper(
        self, batch, state_iters, losses
    ):  # pylint: disable=arguments-renamed
        """Tried to do white background for the paper."""
        idx = slice(0, min(8, batch['hmc_cam_img'].shape[0]))
        batch = slice_batch(batch, idx, ["hmc_distmodel", "camera"])
        state_iters = [s[idx] for s in state_iters]
        # fs = 1  # font scale

        if 'state' not in batch['rosetta_correspondences']:
            batch = self.preprocess_batch_and_rosetta_gt(batch)

        _, N = batch['hmc_cam_img'].shape[:2]
        rosetta_state: State = batch['rosetta_correspondences']['state']
        if rosetta_state.img is None:
            rosetta_state = self.fk(batch, rosetta_state.rtvec, rosetta_state.expr, render=True)
        det_kp2d = batch['hmc_keypoints']

        # ------------------------------------
        # ugly thing to do
        # ------------------------------------
        self.add_front_image(batch, [rosetta_state] + state_iters)
        assert self.decoder.apply_mask
        self.decoder.apply_mask = False

        def doit(state):
            r_mask = state.mask
            r_front_mask = state.front_mask
            state = self.fk(batch, state.rtvec, state.expr)
            self.add_front_image(batch, [state])
            state.mask = r_mask
            state.front_mask = r_front_mask
            return state

        rosetta_state = doit(rosetta_state)
        state_iters = [doit(s) for s in state_iters]
        self.decoder.apply_mask = True
        # ------------------------------------

        rosetta_state.img = [
            rosetta_state.img[:, i].flip(-1) if i % 2 else rosetta_state.img[:, i] for i in range(N)
        ]
        rosetta_state.img = th.stack(rosetta_state.img, 1)
        rosetta_state.front_img = rosetta_state.front_img[..., : -300 // 2, 150 // 2 :]
        rosetta_state.front_mask = rosetta_state.front_mask[..., : -300 // 2, 150 // 2 :]
        r_diag, r_fdiag = self.visualize_state(batch, rosetta_state, 'groundtruth', det_kp2d)

        state_diags = []
        state_fdiags = []
        for i, state in enumerate(state_iters):
            if i not in [0, 1, 2, len(state_iters) - 1]:
                continue

            state.img = [state.img[:, j].flip(-1) if j % 2 else state.img[:, j] for j in range(N)]
            state.img = th.stack(state.img, 1)
            state.front_img = state.front_img[..., : -300 // 2, 150 // 2 :]
            state.front_mask = state.front_mask[..., : -300 // 2, 150 // 2 :]

            state_diag, state_fdiag = self.visualize_state(
                batch, state, f'iter {i}' if i else 'init', det_kp2d, rosetta_state
            )
            state_diags.append(state_diag)
            state_fdiags.append(state_fdiag)

        assert not self.viz_keypoints
        det_kp2d_img = batch['hmc_cam_img'].expand(-1, -1, 3, -1, -1)
        assert 'st_img_check' not in batch
        det_kp2d_img = th.cat(
            [
                th.cat([det_kp2d_img[:, 0], det_kp2d_img[:, 1].flip(-1)], -1),
                th.cat([det_kp2d_img[:, 2], det_kp2d_img[:, 3].flip(-1)], -1),
            ],
            -2,
        )
        det_kp2d_img = det_kp2d_img.permute(0, 2, 3, 1)  # BHWC
        # det_kp2d_img = put_text_on_img(det_kp2d_img.contiguous(), 'hmc input', fs)  # BH(NW)C

        state_diags = pad_and_cat(
            [pad_and_cat([det_kp2d_img, r_diag], 1)] + state_diags, 2
        )  # B(XH)(NW)C

        # BH(NW)C -> BHNWC -> BNHWC -> BV2HWC -> BVH2WC -> B(VH)(2W)C
        det_kp2d_img_grid = det_kp2d_img
        state_fdiags = pad_and_cat(state_fdiags + [r_fdiag, det_kp2d_img_grid], 2)  # B(2H)(XW)C

        ret = {
            'hmc_front': state_diags,
            'front': state_fdiags,
        }

        ret.update(paper_diag_input(batch))
        for i, state in enumerate(state_iters):
            if i in [0, 1, len(state_iters) - 1]:
                ret.update(paper_diag_state(state, f'iter{i}_'))

        return ret
