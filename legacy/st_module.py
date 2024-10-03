import time
from typing import Any, Dict, Optional, Sequence
import torch as th
import torch.nn.functional as thf
from drtk.renderlayer.renderlayer import RenderLayer as RenderLayer
from drtk.renderlayer.projection import project_points

# import IPython  # pylint: disable=unused-import

from utils import (
    put_text_on_img,
    gather_camera_params,
    load_rosetta_stats,
    resize2d,
)
from sdm_utils import (
    State,
    splitbatch_decoder_forward,
    preprocess_batch,
    reshape_for_viz,
    grid_reshape_for_viz,
    # paper_diag_style_transfer,
    # color_correct_2,
    color_correct,
)

from model.tex_conditioning import TexCond
from model.st_model_unet import STCnnUVMap, STCnnRGB
from model.st_model_swin import STSwinUVMap, STSwinRGB
from randaugment import MyAugment
from keypoint import KeypointHandler


class ST_Module(base_classes.Module):
    def __init__(self, decoder, **kwargs) -> None:
        super().__init__()
        self.decoder = decoder
        print(kwargs)
        self.cameras = kwargs.get('cameras')

        # ------------------------------------
        # Keypoint
        # ------------------------------------
        self.kp_handler = KeypointHandler(cameras=self.cameras, **kwargs.get('keypoint_kwargs'))

        # ------------------------------------
        # Configuration
        # ------------------------------------
        self.render_size = tuple(kwargs.get('render_size'))
        self.rotdim = kwargs.get("rotdim", 6)
        self.cond_type = kwargs.get("cond_type")
        self.rgb_loss_type = kwargs.get("rgb_loss_type")
        self.use_gt_uv = kwargs.get("use_gt_uv", False)
        self.init_noise = kwargs.get("init_noise")
        assert self.rgb_loss_type in ['img', 'mimg']
        self.isMouthView = [('mouth' in c) for c in self.cameras]
        assert self.init_noise in [True, False]

        # ------------------------------------
        # Model
        # ------------------------------------
        self.model_type = kwargs.get('model_type')
        self.set_model(**kwargs)
        if self.model:
            print(self.model)
            print("number of parameters", sum(p.numel() for p in self.model.parameters()))

        # ------------------------------------
        # Checkpoint
        # ------------------------------------
        if kwargs.get('ckpt_path') is not None:
            print("Loading ", kwargs['ckpt_path'])
            dct = th.load(kwargs['ckpt_path'], map_location='cpu')
            dct = {k.replace('model.st_model.', ''): v for k, v in dct.items()}
            dct = {k.replace('model.', '', 1): v for k, v in dct.items()}
            ckpt_nc = dct['rgb_model.tex_cnn.conv_in.weight'].shape[1]
            model_nc = self.model.rgb_model.tex_cnn.conv_in.weight.shape[1]
            if model_nc < ckpt_nc:
                print("Stripping tex cnn stem of loaded checkpoint")
                w = dct['rgb_model.tex_cnn.conv_in.weight'][:, :model_nc]
                dct['rgb_model.tex_cnn.conv_in.weight'] = w
            self.model.load_state_dict(dct, strict=True)

        # ------------------------------------
        # Augmentation
        # ------------------------------------
        self.aug_module = None
        image_aug = kwargs.get("image_aug")
        if image_aug != "none":
            print("Using image augmentation", image_aug)
            aug_scale = int(image_aug.split("_")[1])
            self.aug_module = MyAugment(aug_scale)

        # ------------------------------------
        # Misc
        # ------------------------------------
        self.register_buffer(
            "color_map", th.from_numpy(get_color_map("COLORMAP_JET")).byte(), persistent=False
        )
        self.rosetta_keypoints = True
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
        self.front_view_generator = FixedFrontalViewGenerator()

        # ------------------------------------
        # Conditioning for style transfer
        # ------------------------------------
        self.tex_cond = TexCond(decoder, **kwargs)
        self.tex_cond.eval()

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
        self.track_time = False

    def set_model(self, **kwargs):
        assert self.model_type is not None
        self.model = th.nn.Module()
        self.model.uv_model = None
        self.model.rgb_model = None
        # uv model
        if 'uvm_conv' in self.model_type:
            self.model.uv_model = STCnnUVMap(**kwargs)
        elif 'uvm_swin' in self.model_type:
            self.model.uv_model = STSwinUVMap(**kwargs)

        # load pretrained uv model
        if kwargs.get('uvpred_path') is not None:
            print("Loading pretrained uv model from", kwargs['uvpred_path'])
            dct = th.load(kwargs['uvpred_path'], map_location='cpu')
            dct = {k.replace('model.uv_model.', ''): v for k, v in dct.items()}
            self.model.uv_model.load_state_dict(dct, strict=True)

        # freeze uv model if needed
        self.freeze_uvpred = kwargs.get('freeze_uvpred')
        if self.freeze_uvpred:
            print("Freezing uv model")
            for param in self.model.uv_model.parameters():
                param.requires_grad = False
            self.model.uv_model.requires_grad_(False)
            self.model.uv_model.eval()
            self.model.uv_model.apply(freeze)

        # load rgb model
        if 'rgb_cnn' in self.model_type:
            self.model.rgb_model = STCnnRGB(**kwargs)
        elif 'rgb_swin' in self.model_type:
            self.model.rgb_model = STSwinRGB(**kwargs)

        assert self.model.uv_model is not None or self.model.rgb_model is not None

    def train(self, mode=True):
        super().train(mode)
        if self.model.uv_model is not None:
            self.model.uv_model.train(mode and not self.freeze_uvpred)

    def get_optim_params(self, base_lr: float):
        return [{"name": "sdm", "lr": base_lr, "params": self.model.parameters()}]

    def fk(
        self, batch, rtvec, expr, use_distort=False, render=True, onepass=False, get_uv_img=False
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
        # else:
        #     if not self.predict_expr and ('state' in batch['rosetta_correspondences']):
        #         verts_a = batch['rosetta_correspondences']['state'].verts_a
        #     else:
        #         verts_a = self.decoder.forward_geo(batch, expr)

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
            # verts_a=verts_a,
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

    def uv_forward(self, batch):
        raise AttributeError("Don't use uv prediction as of now")
        B, N = batch['hmc_cam_img'].shape[:2]
        batch = self.preprocess_batch_and_rosetta_gt(batch)
        inp_img = batch['hmc_cam_img']
        if self.use_gt_uv:
            pred_mask_logits = batch['rosetta_correspondences']['state'].mask
            pred_mask_logits = pred_mask_logits * 10 - 5
            return {
                'inp_img_uv': inp_img,
                'pred_uv': batch['rosetta_correspondences']['state'].uv_img,
                'pred_mask_logits': pred_mask_logits,
            }
        if self.aug_module is not None and self.training and not self.freeze_uvpred:
            inp_img = inp_img.expand(-1, -1, 3, -1, -1)
            inp_img = self.aug_module.augment_one_img(inp_img.flatten(0, 1)).unflatten(0, (B, N))
            inp_img = inp_img[:, :, :1]
        pred_uv, pred_mask_logits = self.model.uv_model.forward(inp_img, self.cameras)

        outputs = {
            'pred_uv': pred_uv,  # BN2HW
            'pred_mask_logits': pred_mask_logits,  # BN1HW
            'inp_img_uv': inp_img,  # BN1HW
        }

        return outputs

    def rgb_forward(self, batch, uv_outputs=None):
        # B, N = batch['hmc_cam_img'].shape[:2]
        if "results_m1" in batch:
            rtvec = batch["results_m1"]["rtvec"][:, -1]
            expr = batch["results_m1"]["expr"][:, -1]
            if self.training and self.init_noise:
                rtvec = rtvec + th.randn_like(rtvec) * self.std_rtvec * 0.15
                expr = expr + th.randn_like(expr) * self.std_expr * 0.3
            batch["sdm_rtvec"] = rtvec
            batch["sdm_expr"] = expr

        batch = self.preprocess_batch_and_rosetta_gt(batch)
        start_time = time.time()
        batch = self.tex_cond.get_conditioning(batch)  # BM3HW
        inp_img = batch['hmc_cam_img']
        rosetta_state: State = batch['rosetta_correspondences']['state']

        assert '_uv_' not in self.cond_type and 'conduv' in self.cond_type

        cond_uv_img = resize2d(batch['id_cond']['uv_img'], self.render_size)
        cond_mask = resize2d(batch['id_cond']['mask'], self.render_size)
        cond_img = resize2d(batch['id_cond']['img'], self.render_size)
        gt_img = rosetta_state.img
        gt_mask = rosetta_state.mask

        if self.aug_module is not None and self.training:
            ret = self.aug_module.augment_st(
                inp_img, gt_img, cond_img, gt_mask, cond_mask, cond_uv_img
            )
            inp_img, gt_img, cond_img, gt_mask, cond_mask, cond_uv_img = ret
        id_cond = th.cat([cond_img, cond_uv_img * 127 + 127, cond_mask * 255], -3)

        pred_rgb, id_cond = self.model.rgb_model.forward(inp_img, id_cond, uv_outputs)
        taken_time = time.time() - start_time
        if self.track_time:
            print("rgb forward time", taken_time)

        outputs = {'inp_img_rgb': inp_img, 'pred_rgb': pred_rgb}
        outputs.update({'id_cond': cond_img, 'gt_mask': gt_mask, 'gt_img': gt_img})
        return outputs

    def uv_compute_losses(self, batch, outputs):
        rosetta_state: State = batch['rosetta_correspondences']['state']
        losses = {}
        # uv loss
        diff = thf.smooth_l1_loss(
            outputs['pred_uv'] * 128, rosetta_state.uv_img * 128, reduction="none"
        )
        gt_mask = rosetta_state.mask
        uv_loss = (diff * gt_mask).mean((1, 2, 3, 4)) / gt_mask.mean((1, 2, 3, 4))
        losses['uv_l2'] = uv_loss

        diff = thf.binary_cross_entropy_with_logits(
            outputs['pred_mask_logits'], rosetta_state.mask, reduction="none"
        )
        mask_loss = diff.mean((1, 2, 3, 4))
        losses['mask_bce'] = mask_loss

        losses['total'] = losses['uv_l2'] + losses['mask_bce']

        return losses

    def rgb_compute_losses(self, batch, outputs):
        # rosetta_state: State = batch['rosetta_correspondences']['state']
        losses = {}

        # diff = thf.smooth_l1_loss(outputs['pred_rgb'], rosetta_state.img, reduction="none")
        diff = thf.smooth_l1_loss(outputs['pred_rgb'], outputs['gt_img'], reduction="none")

        losses['img_l1'] = diff.mean((1, 2, 3, 4))

        # mask = rosetta_state.mask
        mask = outputs['gt_mask']
        if 'pred_mask_logits' in outputs:
            mask = mask * (outputs['pred_mask_logits'] > 0).float()
        losses['mimg_l1'] = (diff * mask).mean((1, 2, 3, 4)) / (mask.mean((1, 2, 3, 4)) + 1.0e-8)

        losses['total'] = losses[f'{self.rgb_loss_type}_l1']
        return losses

    def uv_get_diag_images(self, batch, outputs, losses):
        rosetta_state: State = batch['rosetta_correspondences']['state']
        fs = 1

        # input image
        inp_img = reshape_for_viz(outputs['inp_img_uv'])  # BH(NW)3
        inp_img = put_text_on_img(inp_img.contiguous(), 'input', fs)

        # uvmap
        udiff = (outputs['pred_uv'] - rosetta_state.uv_img) * rosetta_state.mask
        udiff = udiff.mul(128).abs().mean(-3) * 5
        udiff = reshape_for_viz(self.color_map[udiff.clamp(0, 255).long()])  # BH(NW)3
        gt_uv = reshape_for_viz(rosetta_state.uv_img.add(1).mul(128))  # BH(NW)3
        pred_uv = reshape_for_viz(outputs['pred_uv'].add(1).mul(128))  # BH(NW)3
        gt_uv = put_text_on_img(gt_uv.contiguous(), 'gt uv', fs)
        pred_uv = put_text_on_img(pred_uv.contiguous(), 'pred uv', fs)
        diag_img_uv = th.cat([gt_uv, pred_uv, udiff], 1)  # B(3H)(NW)3

        # mask
        pred_mask = (outputs['pred_mask_logits'] > 0).float()
        mdiff = (pred_mask - rosetta_state.mask).abs().mean(-3) * 5 * 255
        mdiff = reshape_for_viz(self.color_map[mdiff.clamp(0, 255).long()])  # BH(NW)3
        gt_mask_img = reshape_for_viz(rosetta_state.mask.expand(-1, -1, 3, -1, -1) * 255)  # BH(NW)3
        pred_mask = reshape_for_viz(pred_mask.expand(-1, -1, 3, -1, -1) * 255)
        gt_mask_img = put_text_on_img(gt_mask_img.contiguous(), 'gt mask', fs)
        pred_mask = put_text_on_img(pred_mask.contiguous(), 'pred mask', fs)
        diag_img_mask = th.cat([gt_mask_img, pred_mask, mdiff], dim=1)  # B(3H)(NW)3

        # prepare final diag image
        diag_img = pad_and_cat([inp_img, diag_img_uv, diag_img_mask], dim=1)
        return diag_img

    def rgb_get_diag_images(self, batch, outputs, losses):
        # rosetta_state: State = batch['rosetta_correspondences']['state']
        fs = 1

        # input image
        inp_img = reshape_for_viz(outputs['inp_img_rgb'])  # BH(NW)3
        inp_img = put_text_on_img(inp_img.contiguous(), 'input', fs)
        # gt_img = rosetta_state.img
        # gt_mask = rosetta_state.mask
        gt_img = outputs['gt_img']
        gt_mask = outputs['gt_mask']

        if 'pred_mask_logits' in outputs:
            mask = (outputs['pred_mask_logits'] > 0).float()
        else:
            mask = gt_mask

        rdiff = (outputs['pred_rgb'] - gt_img).abs().mean(-3) * 5
        rdiff_masked = (mask * outputs['pred_rgb'] - gt_img).abs().mean(-3) * 5
        rdiff = th.cat([rdiff, rdiff_masked], 2)
        rdiff = reshape_for_viz(self.color_map[rdiff.clamp(0, 255).long()])  # BH(NW)3

        gt_rgb = reshape_for_viz(gt_img)  # BH(NW)3
        gt_rgb = put_text_on_img(gt_rgb.contiguous(), 'gt rgb', fs)

        pred_rgb = outputs['pred_rgb'].clamp(0, 255)
        pred_rgb_masked = mask * outputs['pred_rgb']
        pred_rgb = th.cat([pred_rgb, pred_rgb_masked], 3)
        pred_rgb = reshape_for_viz(pred_rgb)
        pred_rgb = put_text_on_img(pred_rgb.contiguous(), 'pred rgb', fs)

        diag_img_rgb = th.cat([gt_rgb, pred_rgb, rdiff], dim=1)  # B(3H)(NW)3
        diag_img = pad_and_cat([inp_img, diag_img_rgb], dim=1)
        id_cond_img = self.get_id_cond_diag_images(batch, outputs)
        diag_img = pad_and_cat([diag_img, id_cond_img], dim=2)

        return diag_img

    def get_id_cond_diag_images(self, batch, outputs):
        ret = []
        _, N, M = outputs['id_cond'].shape[:3]

        if 'uw_tex' in batch['id_cond']:
            # add id conditioning unwrapped texture and image
            id_cond = resize2d(batch['id_cond']['uw_tex'], None, scale_factor=0.5)
            id_cond = id_cond.transpose(1, 2).flatten(1, 2)  # B(MN)3HW
            id_cond = grid_reshape_for_viz(id_cond, (M, N))  # B(XH)(YW)3
            ret.append(id_cond)

        if 'img' in batch['id_cond']:
            id_cond_img = resize2d(batch['id_cond']['img'], None, scale_factor=0.5)
            id_cond_img = id_cond_img.transpose(1, 2).flatten(1, 2)  # B(MN)3HW
            id_cond_img = grid_reshape_for_viz(id_cond_img, (M, N))  # B(XH)(YW)3
            ret.append(id_cond_img)

        id_cond_inp = outputs['id_cond'].transpose(1, 2).flatten(1, 2)  # B(MN)3HW
        id_cond_inp = grid_reshape_for_viz(id_cond_inp, (M, N))

        ret = [id_cond_inp] + ret
        ret = pad_and_cat(ret, dim=2)

        return ret

    def forward(self, batch, compute_losses=False):  # pylint: disable=arguments-renamed
        # self.get_cond_figure(batch)

        outputs = {}
        if 'uvm_' in self.model_type:
            outputs.update(self.uv_forward(batch))
        if 'rgb_' in self.model_type:
            outputs.update(self.rgb_forward(batch, uv_outputs=outputs))

        if compute_losses:
            losses = self.compute_losses(batch, outputs)
            return outputs, losses
        else:
            return outputs

    def compute_losses(self, batch, outputs):
        losses = {'total': 0}
        if 'uvm_' in self.model_type:
            uv_losses = self.uv_compute_losses(batch, outputs)
            losses['total'] += uv_losses.pop('total')
            losses.update(uv_losses)

        if 'rgb_' in self.model_type:
            rgb_losses = self.rgb_compute_losses(batch, outputs)
            losses['total'] += rgb_losses.pop('total')
            losses.update(rgb_losses)
        return losses

    def get_diag_images(self, batch, outputs, losses):  # pylint: disable=arguments-renamed
        diag_images = []
        if 'uvm_' in self.model_type:
            uv_diag_images = self.uv_get_diag_images(batch, outputs, losses)
            diag_images.append(uv_diag_images)

        if 'rgb_' in self.model_type:
            rgb_diag_images = self.rgb_get_diag_images(batch, outputs, losses)
            diag_images.append(rgb_diag_images)

        diag_images = pad_and_cat(diag_images, dim=2)

        # ret = {'st': diag_images}
        # ret.update(paper_diag_style_transfer(batch, outputs))
        # return ret
        return {'st': diag_images}

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

    def get_cond_figure(self, batch):
        """To generate conditioning expressions figure for paper."""
        B = batch['hmc_cam_img'].shape[0]
        for idx in range(4):
            expr = self.tex_cond.cond_codes[idx][None].expand(B, -1)
            rtvec = self.mean_rtvec[None].expand(B, -1)

            state = self.fk(batch, rtvec, expr)
            self.add_front_image(batch, [state])
            img = state.img
            fimg = state.front_img

            # img = color_correct_2(img, 2)
            img = color_correct(img.flatten(0, 1)).unflatten(0, (B, 4))
            img = th.cat(
                [
                    th.cat([img[:, 0], img[:, 1].flip(-1)], -1),
                    th.cat([img[:, 2], img[:, 3].flip(-1)], -1),
                ],
                -2,
            )  # B3H'W'
            img = img.permute(0, 2, 3, 1).byte().cpu().numpy()  # BH'W'3

            # fimg = color_correct_2(fimg, 1)
            fimg = fimg[:, :, : -300 // 2, 150 // 2 :]
            fimg = color_correct(fimg)
            fimg = fimg.permute(0, 2, 3, 1).byte().cpu().numpy()  # BH'W'3

            names = batch["index"]["vrs_file"]
            out_dir = "/mnt/home/chpatel/CARE/runs/cond_expr_figure/outputs"

            for name, im, fim in zip(names, img, fimg):
                name = name[:6]
                typed.save(f"{out_dir}/{name}_{idx}_hmc.png", im)
                typed.save(f"{out_dir}/{name}_{idx}_front.png", fim)

    def compute_eval(self, batch, outputs):
        """For Shaojie's evaluation script. Used by sdm_eval_m2 stage."""
        diff = thf.smooth_l1_loss(outputs['pred_rgb'], outputs['gt_img'], reduction="none")
        image_L1 = diff.mean((1, 2, 3, 4))

        mask = outputs['gt_mask']
        if 'pred_mask_logits' in outputs:
            mask = mask * (outputs['pred_mask_logits'] > 0).float()
        masked_image_L1 = (diff * mask).mean((1, 2, 3, 4)) / (mask.mean((1, 2, 3, 4)) + 1.0e-8)

        image_L1 = image_L1.detach().cpu().numpy().tolist()
        masked_image_L1 = masked_image_L1.detach().cpu().numpy().tolist()
        ret = [
            {
                'image_L1': image_L1[i],
                'masked_image_L1': masked_image_L1[i],
            }
            for i in range(len(image_L1))
        ]
        return ret


class ST_ModuleBuilder(base_classes.ModuleBuilder):
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs: Dict[str, Any] = kwargs

    # pyre-ignore[14]: Inconsistent override
    def build_module(
        self,
        stage: Optional["Stage"] = None,
        dependencies: Optional[Dict[str, "Stage"]] = None,
        nets: Optional[Dict[str, th.nn.Module]] = None,
        stream_list: Optional[Sequence[str]] = None,
    ) -> th.nn.Module:
        # update number of views from capture config
        self.kwargs['cameras'] = stage.config.capture_configs.hmc_capture.cameras
        self.kwargs = {k: v for k, v in self.kwargs.items()}
        return ST_Module(decoder=nets['decoder'], **self.kwargs)
