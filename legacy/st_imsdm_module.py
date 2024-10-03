from typing import Any, Dict, Optional, Sequence
import copy
import torch as th

# import IPython  # pylint: disable=unused-import

from imsdm_module import IMSDM_Module
from st_module import ST_Module
from utils import resize2d


class ST_IMSDM_Module(base_classes.Module):
    def __init__(self, decoder, **kwargs) -> None:
        super().__init__()
        self.decoder = decoder
        print(kwargs)

        self.freeze_sdm = kwargs.get('freeze_sdm')
        self.freeze_st = kwargs.get('freeze_st')

        self.set_model(**kwargs)

    def set_model(self, **kwargs):
        self.model = th.nn.Module()
        self.model.sdm_model = IMSDM_Module(decoder=self.decoder, **kwargs['sdm_kwargs'])
        self.model.st_model = ST_Module(decoder=self.decoder, **kwargs['st_kwargs'])

        if kwargs.get('sdm_ckpt_path') is not None:
            print("Loading pretrained sdm model from", kwargs['sdm_ckpt_path'])
            dct = th.load(kwargs['sdm_ckpt_path'], map_location='cpu')
            missing, unexpected = self.model.sdm_model.load_state_dict(dct, strict=False)
            assert len(unexpected) == 0, f"Unexpected keys in pretrained sdm model: {unexpected}"
            assert all(
                m.startswith('decoder.') for m in missing
            ), f"Missing keys in pretrained sdm model: {missing}"

        if kwargs.get('st_ckpt_path') is not None:
            print("Loading pretrained st model from", kwargs['st_ckpt_path'])
            dct = th.load(kwargs['st_ckpt_path'], map_location='cpu')
            dct = {k.replace('model.st_model.', ''): v for k, v in dct.items()}
            missing, unexpected = self.model.st_model.load_state_dict(dct, strict=False)
            assert len(unexpected) == 0, f"Unexpected keys in pretrained st model: {unexpected}"
            assert all(
                m.startswith('decoder.')
                or m.startswith('tex_cond.')
                or 'relative_position_index' in m
                or '.attn_mask' in m
                for m in missing
            ), f"Missing keys in pretrained st model: {missing}"

        if self.freeze_sdm:
            print("Freezing sdm model")
            for param in self.model.sdm_model.parameters():
                param.requires_grad = False
            self.model.sdm_model.requires_grad_(False)
            self.model.sdm_model.eval()
            self.model.sdm_model.apply(freeze)

        if self.freeze_st:
            print("Freezing st model")
            for param in self.model.st_model.parameters():
                param.requires_grad = False
            self.model.st_model.requires_grad_(False)
            self.model.st_model.eval()
            self.model.st_model.apply(freeze)

    def train(self, mode=True):
        super().train(mode)
        self.model.sdm_model.train(mode and not self.freeze_sdm)
        self.model.st_model.train(mode and not self.freeze_st)

    def forward(self, batch, compute_losses=False):  # pylint: disable=arguments-renamed
        prev_rtvec, prev_expr = None, None

        for key in ['results_m3', 'results_m2', 'results_m1']:
            if key in batch:
                prev_rtvec = batch[key]['rtvec'][:, -1]
                prev_expr = batch[key]['expr'][:, -1]
                batch.pop(key)
                break
        if prev_rtvec is None:
            init_state = self.model.sdm_model.get_init_state(batch)
            prev_rtvec = init_state.rtvec
            prev_expr = init_state.expr

        batch_orig = copy.deepcopy(batch)

        outputs = []
        st_losses = None
        sdm_losses = None
        assert not th.is_grad_enabled()

        for _ in range(2):
            batch = copy.deepcopy(batch_orig)
            batch['sdm_rtvec'] = prev_rtvec
            batch['sdm_expr'] = prev_expr
            st_outputs = self.model.st_model.forward(batch, compute_losses=compute_losses)
            if compute_losses:
                st_outputs, st_losses = st_outputs
            outputs.append(st_outputs)
            prev_st_img = st_outputs['pred_rgb']

            batch = copy.deepcopy(batch_orig)
            batch['st_img_check'] = resize2d(prev_st_img, self.model.sdm_model.render_size)
            batch['rtvec_init'] = prev_rtvec
            batch['expr_init'] = prev_expr
            sdm_outputs = self.model.sdm_model.forward(
                batch, compute_losses=compute_losses, num_iters=3
            )
            if compute_losses:
                sdm_outputs, sdm_losses = sdm_outputs
            outputs.append(sdm_outputs)
            prev_rtvec = sdm_outputs[-1].rtvec
            prev_expr = sdm_outputs[-1].expr

        if compute_losses:
            return outputs, (st_losses, sdm_losses)
        else:
            return outputs

    def get_diag_images(self, batch, outputs, losses):  # pylint: disable=arguments-renamed
        batch['id_cond'] = {}
        batch_orig = copy.deepcopy(batch)

        diag_images = {}
        for i in range(len(outputs) // 2):
            batch = copy.deepcopy(batch_orig)
            batch = self.model.st_model.preprocess_batch_and_rosetta_gt(batch)
            dimg = self.model.st_model.get_diag_images(batch, outputs[2 * i], None)
            diag_images.update({f'r{2*i}_{k}': v for k, v in dimg.items()})
            prev_st_img = outputs[2 * i]['pred_rgb']

            batch = copy.deepcopy(batch_orig)
            batch = self.model.sdm_model.preprocess_batch_and_rosetta_gt(batch)
            batch['st_img_check'] = resize2d(prev_st_img, self.model.sdm_model.render_size)
            dimg = self.model.sdm_model.get_diag_images(batch, outputs[2 * i + 1], None)
            diag_images.update({f'r{2*i+1}_{k}': v for k, v in dimg.items()})

        return diag_images

    def compute_eval(self, batch, outputs):
        batch['id_cond'] = {}
        batch_orig = copy.deepcopy(batch)

        evals = []
        for i in range(len(outputs) // 2):
            batch = copy.deepcopy(batch_orig)
            batch = self.model.st_model.preprocess_batch_and_rosetta_gt(batch)
            eval = self.model.st_model.compute_eval(batch, outputs[2 * i])
            evals.append(eval)
            prev_st_img = outputs[2 * i]['pred_rgb']

            batch = copy.deepcopy(batch_orig)
            batch = self.model.sdm_model.preprocess_batch_and_rosetta_gt(batch)
            batch['st_img_check'] = resize2d(prev_st_img, self.model.sdm_model.render_size)
            eval = self.model.sdm_model.compute_eval(batch, outputs[2 * i + 1])
            evals.append(eval)

        return evals


class ST_IMSDM_ModuleBuilder(base_classes.ModuleBuilder):
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
        self.kwargs['sdm_kwargs']['cameras'] = stage.config.capture_configs.hmc_capture.cameras
        self.kwargs['st_kwargs']['cameras'] = stage.config.capture_configs.hmc_capture.cameras
        self.kwargs = {k: v for k, v in self.kwargs.items()}
        return ST_IMSDM_Module(decoder=nets['decoder'], **self.kwargs)
