from typing import Any, Dict, Optional, Sequence
import torch as th

# import IPython  # pylint: disable=unused-import

from sdm_utils import State
from utils import get_num_iters
from sdm_module import SDM_Module
from model.imsdm_model import (
    ImageSDMTransformer,
)
from model.imsdm_cnn_model import ImageSDMCNNMLP
from randaugment import MyAugment
from model.core import TwoModelWrapper


class IMSDM_Module(SDM_Module):
    def __init__(self, decoder, **kwargs) -> None:
        super().__init__(decoder, **kwargs)
        self.data_feature_type = kwargs.get("data_feature_type")
        self.mask_st_img = kwargs.get("mask_st_img")
        assert self.data_feature_type in [
            'rimg_img',
            'res_rimg_img',
            'ir_img',
            'ir3_img',
            'stimg_img',
            'stimg_img_ir',
        ]
        self.viz_keypoints = False

        self.normalize_input_image = kwargs.get("normalize_input_image")
        if self.normalize_input_image:
            print("NORMALIZING INPUT IMAGE")

        self.aug_module = None
        image_aug = kwargs.get("image_aug")
        if image_aug != "none":
            print("Using image augmentation", image_aug)
            aug_scale = int(image_aug.split("_")[1])
            self.aug_module = MyAugment(aug_scale)

    def set_model(self, **kwargs):
        two_models = kwargs.get("two_models")
        model_type = kwargs.get("model_type")
        model_cls = {
            "v1": ImageSDMTransformer,
            "cnnmlp": ImageSDMCNNMLP,
        }[model_type]
        if two_models:
            self.model = TwoModelWrapper(
                model_cls, 1, self.num_pred_params, self.kp_handler, **kwargs
            )
        else:
            self.model = model_cls(self.num_pred_params, self.kp_handler, **kwargs)

    def fk(self, batch, rtvec, expr, use_distort=False, render=True, get_uv_img=False):
        # Changed default render=False to render=True
        return super().fk(
            batch,
            rtvec,
            expr,
            use_distort=use_distort,
            render=render,
            onepass=True,
            get_uv_img=get_uv_img,
        )

    def fk_experimental(self, batch, rtvec, expr, use_distort=False, render=True):
        # # Changed default render=False to render=True
        # # return super().fk(batch, rtvec, expr, use_distort=use_distort, render=render, onepass=True)
        # if use_distort or not render or 'state' not in batch['rosetta_correspondences']:
        #     return super().fk(
        #         batch, rtvec, expr, use_distort=use_distort, render=render, onepass=True
        #     )
        # with th.set_grad_enabled(True):
        #     if rtvec.requires_grad:
        #         rtvec.retain_grad()
        #     if expr.requires_grad:
        #         expr.retain_grad()
        #     rtvec.requires_grad_(self.predict_slope)
        #     expr.requires_grad_(self.predict_expr)
        #     state = super().fk(
        #         batch, rtvec, expr, use_distort=use_distort, render=render, onepass=True
        #     )

        #     rosetta_state: State = batch['rosetta_correspondences']['state']
        #     diff = rosetta_state.img - state.img
        #     diff.abs().mean().backward(retain_graph=True, inputs=[rtvec, expr])

        #     grds = [rtvec.grad.detach()] if self.predict_slope else []
        #     grds += [expr.grad.detach()] if self.predict_expr else []
        #     gfeat = th.cat(grds, -1)
        # state.gfeat = gfeat
        # return state
        pass

    def guiding_feature(self, batch, state: State):
        if not self.input_guiding_feature:
            B, device, dtype = state.rtvec.shape[0], state.rtvec.device, state.rtvec.dtype
            return th.zeros(B, self.num_pred_params, device=device, dtype=dtype)
        rosetta_state: State = batch['rosetta_correspondences']['state']
        with th.set_grad_enabled(True):
            rtvec_ff = state.rtvec.detach().clone()
            expr_ff = state.expr.detach().clone()
            rtvec_ff.requires_grad_(self.predict_slope)
            expr_ff.requires_grad_(self.predict_expr)

            state_ff = self.fk(batch, rtvec_ff, expr_ff, render=True)

            diff = rosetta_state.img - state_ff.img
            diff.abs().mean().backward(retain_graph=False)

            grds = [rtvec_ff.grad] if self.predict_slope else []
            grds += [expr_ff.grad] if self.predict_expr else []
            gfeat = th.cat(grds, -1)
        return gfeat

    def data_feature(self, batch, state: State):
        B, N = batch['hmc_cam_img'].shape[:2]
        if not self.input_residual:
            device, dtype = state.rtvec.device, state.rtvec.dtype
            h, w = self.render_size
            c = {
                'rimg_img': 6,
                'res_rimg_img': 3,
                'ir_img': 4,
                'ir3_img': 6,
                'stimg_img': 6,
                'stimg_img_ir': 7,
            }[self.data_feature_type]
            return th.zeros(B, N, c, h, w, device=device, dtype=dtype)
        rosetta_state: State = batch['rosetta_correspondences']['state']
        rimg = rosetta_state.img
        img = state.img
        hmc_img = batch['hmc_cam_img'].float()

        if self.data_feature_type in ['stimg_img', 'stimg_img_ir']:
            rimg = batch['st_img_check']
            if self.mask_st_img:
                mask = state.mask
                mask = mask * batch['st_mask_check'] if 'st_mask_check' in batch else mask

                rimg = rimg * mask
                img = img * mask

        if self.aug_module is not None and self.training:
            # raise AttributeError("augmentation shouldn't be used for export")
            rimg, img = self.aug_module.augment_hmc(rimg, img)
            if 'hmc' in self.data_feature_type:
                hmc_img = self.aug_module.augment_one_img(hmc_img.flatten(0, 1))
                hmc_img = hmc_img.unflatten(0, (B, N))

        if self.data_feature_type == 'rimg_img':
            dfeat = th.cat([rimg, img], -3)  # BN6HW
        elif self.data_feature_type == 'res_rimg_img':
            dfeat = rimg - img  # BN3HW
        elif self.data_feature_type == 'ir_img':
            dfeat = th.cat([hmc_img, img], -3)
        elif self.data_feature_type == 'ir3_img':
            dfeat = th.cat([hmc_img.expand(-1, -1, 3, -1, -1), img], -3)
        elif self.data_feature_type == 'stimg_img':
            dfeat = th.cat([rimg, img], -3)  # BN6HW
        elif self.data_feature_type == 'stimg_img_ir':
            dfeat = th.cat([rimg, img, hmc_img], -3)  # BN7HW
        else:
            raise NotImplementedError

        if self.normalize_input_image:
            dfeat = dfeat * (2 / 255) - 1
        return dfeat


class IMSDM_Module_Alternate(IMSDM_Module):
    """Tried this once but didn't work any better. Ignore as of now."""

    def __init__(self, decoder, **kwargs) -> None:
        super().__init__(decoder, **kwargs)
        print("ALTERNATING BETWEEN SLOPE AND EXPRESSION MODELS")

    def set_model(self, **kwargs):
        two_models = kwargs.get("two_models")
        model_type = kwargs.get("model_type")
        model_cls = {"v1": ImageSDMTransformer}[model_type]
        assert not two_models

        assert kwargs.get('predict_expression')
        assert kwargs.get('predict_slope')
        slope_kwargs = kwargs.copy()
        slope_kwargs['predict_expression'] = False
        expr_kwargs = kwargs.copy()
        expr_kwargs['predict_slope'] = False

        self.slope_model = model_cls(3 + self.rotdim, self.kp_handler, **kwargs)
        self.expr_model = model_cls(self.expr_dim, self.kp_handler, **kwargs)
        print(self.slope_model)
        print(self.expr_model)
        print("number of parameters", sum(p.numel() for p in self.slope_model.parameters()))
        print("number of parameters", sum(p.numel() for p in self.expr_model.parameters()))
        self.model = None

    def get_optim_params(self, base_lr: float):
        return [
            {
                "name": "sdm",
                "lr": base_lr,
                "params": list(self.slope_model.parameters()) + list(self.expr_model.parameters()),
            }
        ]

    def forward(self, batch, compute_losses=False):  # pylint: disable=arguments-renamed
        batch = self.preprocess_batch_and_rosetta_gt(batch)

        state = self.get_init_state(batch)
        state_iters = [state]

        if self.training:
            num_iters = get_num_iters(batch['iteration'], self.num_iters, self.num_iters_schedule)
        else:
            num_iters = self.num_val_iters
        for iter_idx in range(num_iters * 2):  # pylint: disable=unused-variable
            state = state.detach()
            if iter_idx % 2 == 0:
                model = self.slope_model
                self.predict_expr = False
                self.predict_slope = True
            else:
                model = self.expr_model
                self.predict_expr = True
                self.predict_slope = False
            dfeat = self.data_feature(batch, state)
            sfeat = self.state_feature(batch, state)
            gfeat = self.guiding_feature(batch, state)
            out = model.forward([dfeat, sfeat, gfeat], iter_idx)
            state = self.update_state(batch, state, out)

            state_iters.append(state)

        self.predict_expr = True
        self.predict_slope = True
        if compute_losses:
            losses = self.compute_losses(batch, state_iters)
            return state_iters, losses
        else:
            return state_iters

    def compute_losses(self, batch, state_iters: Sequence[State]):
        bkp_weight = self.loss_cfg.weight
        ltype = self.loss_cfg.type
        todo = [k for k, v in bkp_weight.items() if v > 1.0e-9]

        for k in todo:
            assert k in ['img', 'fimg', 'kp3d']
        slope_iters = [state_iters[0]] + state_iters[1::2]
        expr_iters = [state_iters[0]] + state_iters[2::2]
        assert len(slope_iters) == len(expr_iters)

        total_loss = 0
        losses = {}
        for lt, st in [
            ('img', state_iters),
            ('fimg', expr_iters),
            ('kp3d', slope_iters),
        ]:
            if lt not in todo:
                continue
            self.loss_cfg.weight = {lt: bkp_weight[lt]}
            temp = super().compute_losses(batch, st, get_metrics=False)
            total_loss += temp['total']
            k = f'{lt}_{ltype[lt]}'
            losses[k] = temp[k]

        losses['total'] = total_loss
        self.loss_cfg.weight = bkp_weight

        with th.no_grad():
            metrics = self.compute_metrics(batch, state_iters)
        losses.update(metrics)
        return losses


class IMSDM_ModuleBuilder(base_classes.ModuleBuilder):
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
        if self.kwargs.get('alternate'):
            return IMSDM_Module_Alternate(decoder=nets['decoder'], **self.kwargs)
        self.kwargs = {k: v for k, v in self.kwargs.items()}
        return IMSDM_Module(decoder=nets['decoder'], **self.kwargs)
