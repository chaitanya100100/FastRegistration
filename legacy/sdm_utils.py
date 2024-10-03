from dataclasses import dataclass
import torch
from .utils import (
    jac_reproj_error_wrt_kp2d,
    jac_kp2d_wrt_campos_camrot,
    jac_camrotpos_wrt_rtvec,
    gather_camera_params,
    undistort_hmc_keypoints_batch,
    rotmat_to_rot6d,
    undistort_hmc_cam_img_batch,
    resize2d,
)
from .keypoint import KeypointHandler


@dataclass
class State:
    rtvec: torch.Tensor  # B x [rotdim + 3]
    expr: torch.Tensor  # B x 256
    verts_a: torch.Tensor = None  # B x V x 3
    kp3d_a: torch.Tensor = None  # B x K x 3
    verts: torch.Tensor = None  # B x N x V x 3
    kp3d: torch.Tensor = None  # B x N x K x 3
    kp2d: torch.Tensor = None  # B x N x K x 3
    img: torch.Tensor = None  # B x N x 3 x H x W
    mask: torch.Tensor = None  # B x N x 1 x H x W
    front_img: torch.Tensor = None  # B x 3 x H x W
    front_mask: torch.Tensor = None  # B x 1 x H x W
    use_distort: bool = False
    uv_img: torch.Tensor = None  # B x N x 2 x H x W

    def detach(self):
        return State(
            rtvec=self.rtvec.detach(),
            expr=self.expr.detach(),
            verts_a=None if self.verts_a is None else self.verts_a.detach(),
            kp3d_a=None if self.kp3d_a is None else self.kp3d_a.detach(),
            verts=None if self.verts is None else self.verts.detach(),
            kp3d=None if self.kp3d is None else self.kp3d.detach(),
            kp2d=None if self.kp2d is None else self.kp2d.detach(),
            img=None if self.img is None else self.img.detach(),
            mask=None if self.mask is None else self.mask.detach(),
            front_img=None if self.front_img is None else self.front_img.detach(),
            front_mask=None if self.front_mask is None else self.front_mask.detach(),
            use_distort=self.use_distort,
            uv_img=None if self.uv_img is None else self.uv_img.detach(),
        )

    def __len__(self):
        return len(self.rtvec)

    def __getitem__(self, idx):
        return State(
            rtvec=self.rtvec[idx],
            expr=self.expr[idx],
            verts_a=None if self.verts_a is None else self.verts_a[idx],
            kp3d_a=None if self.kp3d_a is None else self.kp3d_a[idx],
            verts=None if self.verts is None else self.verts[idx],
            kp3d=None if self.kp3d is None else self.kp3d[idx],
            kp2d=None if self.kp2d is None else self.kp2d[idx],
            img=None if self.img is None else self.img[idx],
            mask=None if self.mask is None else self.mask[idx],
            front_img=None if self.front_img is None else self.front_img[idx],
            front_mask=None if self.front_mask is None else self.front_mask[idx],
            use_distort=self.use_distort,
            uv_img=None if self.uv_img is None else self.uv_img[idx],
        )

    def __repr__(self) -> str:
        return (
            f"State(rtvec={self.rtvec.shape}, expr={self.expr.shape}, "
            f"verts_a={self.verts_a.shape if self.verts_a is not None else None}, "
            f"kp3d_a={self.kp3d_a.shape if self.kp3d_a is not None else None}, "
            f"verts={self.verts.shape if self.verts is not None else None}, "
            f"kp3d={self.kp3d.shape if self.kp3d is not None else None}, "
            f"kp2d={self.kp2d.shape if self.kp2d is not None else None}, "
            f"img={self.img.shape if self.img is not None else None}, "
            f"front_img={self.front_img.shape if self.front_img is not None else None}, "
            f"front_mask={self.front_mask.shape if self.front_mask is not None else None}, "
            f"use_distort={self.use_distort})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def resize(self, img_size):
        h, w = img_size
        _, _, _, H, W = self.img.shape
        if H == h and W == w:
            return self

        sc = torch.tensor([h / H, w / W, 1], device=self.img.device)
        return State(
            rtvec=self.rtvec,
            expr=self.expr,
            verts_a=None if self.verts_a is None else self.verts_a,
            kp3d_a=None if self.kp3d_a is None else self.kp3d_a,
            verts=None if self.verts is None else self.verts,
            kp3d=None if self.kp3d is None else self.kp3d,
            kp2d=None if self.kp2d is None else self.kp2d * sc[None, None, None, :],
            img=None if self.img is None else resize2d(self.img, (h, w)),
            mask=None if self.mask is None else resize2d(self.mask, (h, w)),
            front_img=None if self.front_img is None else self.front_img,
            front_mask=None if self.front_mask is None else self.front_mask,
            use_distort=self.use_distort,
            uv_img=None if self.uv_img is None else resize2d(self.uv_img, (h, w)),
        )


def jacobian_repro_wrt_rtvec(
    batch,
    state: State,
    kp_handler: KeypointHandler,
    use_kp_subset: bool,
    img_shape: torch.Tensor,
):
    B, N = batch["hmc_cam_img"].shape[:2]
    gt_kp2d = batch['hmc_keypoints'].view(B * N, -1, 3)
    pred_kp2d = state.kp2d.view(B * N, -1, 3)
    kp3d_a_rep = state.kp3d_a[:, None].expand(-1, N, -1, -1).flatten(0, 1)  # (BN)K3
    rtvec = state.rtvec
    P = rtvec.shape[1]
    if use_kp_subset:
        gt_kp2d = kp_handler.select_subset(gt_kp2d.view(B, N, -1, 3)).view(B * N, -1, 3)
        pred_kp2d = kp_handler.select_subset(pred_kp2d.view(B, N, -1, 3)).view(B * N, -1, 3)
        kp3d_a_rep = kp_handler.select_subset(kp3d_a_rep.view(B, N, -1, 3)).view(B * N, -1, 3)

    # (BN)K3
    jac_err_kp2d = jac_reproj_error_wrt_kp2d(
        pred_kp2d[..., :2], gt_kp2d[..., :2], gt_kp2d[..., 2], 'l1'
    )

    camera_params = gather_camera_params(rtvec, batch['hmc_Rt'], batch['hmc_K'], None)
    # (BN)K23 and (BN)K233
    jac_kp2d_campos, jac_kp2d_camrot = jac_kp2d_wrt_campos_camrot(kp3d_a_rep, **camera_params)
    jac_kp2d_camrotpos = torch.cat([jac_kp2d_camrot, jac_kp2d_campos[..., None]], -1)  # (BN)K234

    # (BN)34P
    jac_camrotpos_rtvec = jac_camrotpos_wrt_rtvec(batch['hmc_Rt'], rtvec)

    # (BN)K2P
    jac_kp2d_rtvec = torch.einsum('BKxij,Bijl->BKxl', jac_kp2d_camrotpos, jac_camrotpos_rtvec)
    # (BN)K2P
    jac_err_rtvec = torch.einsum('BKx,BKxl->BKxl', jac_err_kp2d, jac_kp2d_rtvec)
    # BNK2P
    jac_err_rtvec = jac_err_rtvec.view(B, N, -1, 2, P)

    return jac_err_rtvec / img_shape[..., None]


def preprocess_batch(batch):
    # if only one view then append a view dimensions
    if batch['hmc_K'].dim() == 3:
        for k in [
            'hmc_K',
            'hmc_Rt',
            'hmc_distcoeffs',
            'hmc_cam_img',
        ]:
            batch[k] = batch[k][:, None]

    # get undistorted target keypoints
    batch['hmc_keypoints_original'] = batch['hmc_keypoints']
    batch['hmc_keypoints'] = undistort_hmc_keypoints_batch(
        batch['hmc_keypoints'],
        batch['hmc_K'],
        batch['hmc_distmodel'],
        batch['hmc_distcoeffs'],
        batch['hmc_cam_img'].shape[-2:],
    )

    # get undistorted image
    batch['hmc_cam_img_original'] = batch['hmc_cam_img']
    undist = undistort_hmc_cam_img_batch(
        batch['hmc_cam_img'], batch['hmc_K'], batch['hmc_distmodel'], batch['hmc_distcoeffs']
    )
    batch['hmc_cam_img'] = undist

    # compute rot6d for rosetta avatar2hmc_rvec
    rosetta_gt = batch['rosetta_correspondences']
    rot6d = rotmat_to_rot6d(rvec_to_R(rosetta_gt['avatar2hmc_rvec']))
    batch['rosetta_correspondences']['avatar2hmc_rot6d'] = rot6d
    return batch


def splitbatch_decoder_forward(decoder, ident, expr, camera_params, onepass=False):
    # If not training, then do multiple passes of small batches. This allows us to reduce
    # max GPU memory usage and we can use smaller GPUs with larger batch sizes.
    if torch.is_grad_enabled() or onepass:
        b = expr.shape[0]
    else:
        b = 32

    B = expr.shape[0]
    decoder_out = None
    for st in range(0, B, b):
        en = min(st + b, B)
        cam_b = {k: v[st:en] for k, v in camera_params.items() if k not in ['render_h', 'render_w']}
        cam_b['render_h'] = camera_params['render_h']
        cam_b['render_w'] = camera_params['render_w']

        decb, _ = decoder.forward(
            data={'index': {'ident': ident[st:en]}},
            inputs={
                'expression': expr[st:en],
                'camera_params': cam_b,
            },
            compute_losses=False,
        )

        if decoder_out is None:
            decoder_out = decb
        else:
            for k, v in decb.items():
                decoder_out[k] = torch.cat([decoder_out[k], v], 0)

    return decoder_out


def reshape_for_viz(x):
    # x: BNCHW where C=1,2,3 or BNHWC where C=3
    # returns BH(N*W)C
    if x.shape[-1] == 3 and x.shape[2] != 3:  # channel last
        x = x.permute(0, 1, 4, 2, 3)
    if x.shape[2] == 2:  # for uv
        x = torch.cat([x, torch.zeros_like(x[:, :, :1])], 2)
    if x.shape[2] == 1:  # for hmc
        x = x.expand(-1, -1, 3, -1, -1)
    B, N, _, H, W = x.shape
    return x.permute(0, 3, 1, 4, 2).reshape(B, H, N * W, -1)


def grid_reshape_for_viz(x, grid=None):
    _, N, _, _, _ = x.shape
    if grid is None:
        grid = (N // 2, -1)
    # BN3HW -> BXY3HW -> BXHYW3 -> B(XH)(YW)3
    x = x.unflatten(1, grid).permute(0, 1, 4, 2, 5, 3).flatten(3, 4).flatten(1, 2)
    return x


def pick_rgb_from_uv(uv, prob, tex):
    # uv: BN2HW
    # prob: BN(M+1)HW
    # tex: BM3HW
    B, N = uv.shape[:2]
    M = tex.shape[1]

    prob = prob.flatten(0, 1)  # (BN)(M+1)HW

    tex_bla = tex[:, None].expand(-1, N, -1, -1, -1, -1).flatten(0, 1)  # (BN)M3HW
    uv_bla = uv.permute(0, 1, 3, 4, 2).flatten(0, 1)  # (BN)HW2
    picked = torch.stack(
        [
            torch.nn.functional.grid_sample(tex_bla[:, i], uv_bla, align_corners=False)
            for i in range(M)
        ],
        1,
    )  # (BN)M3HW

    picked_bla = torch.cat([picked, torch.zeros_like(picked[:, :1])], 1)  # (BN)(M+1)3HW
    rgb = (prob[:, :, None] * picked_bla).sum(1)  # (BN)3HW

    return picked.unflatten(0, (B, N)), rgb.unflatten(0, (B, N))


class ColorCorrection(torch.nn.Module):
    def __init__(self):
        super(ColorCorrection, self).__init__()

        self.gamma = 2.0
        self.black = 3.0 / 255.0
        self.register_buffer("scale", torch.FloatTensor([1.6, 1.1, 1.4])[None, :, None, None] / 1.1)

    def forward(self, image):
        # image: NxCxHxW
        # white balance
        image = image * self.scale  # TEMP

        # gamma correction
        image = 0.95 * (1.0 / (1 - self.black)) * (image - self.black).clamp(min=0, max=1)
        image = image.pow(1.0 / self.gamma) - 5.0 / 255.0

        return 255 * image.clamp(min=0, max=1)


def color_correct(image):
    color_correct_func = ColorCorrection().cuda()
    mask = image > 250
    image_out = color_correct_func(image / 255)
    image_out[mask] = 255
    return image_out


def color_correct_2(image, dim):
    # return linear2color_corr(image / 255, dim) * 255
    mask = image > 250
    image_out = linear2color_corr(image / 255, dim) * 255
    image_out[mask] = 255
    return image_out


def paper_diag_state(state: State, prefix=''):
    # fimg = color_correct(state.front_img)  # B3HW
    fimg = linear2color_corr(state.front_img / 255, 1) * 255  # B3HW
    fimg = fimg[:, :, : -300 // 2, 150 // 2 :]

    # img = color_correct(state.img)  # BN3HW
    img = linear2color_corr(state.img / 255, 2) * 255  # BN3HW
    img = torch.cat(
        [
            torch.cat([img[:, 0], img[:, 1].flip(-1)], -1),
            torch.cat([img[:, 2], img[:, 3].flip(-1)], -1),
        ],
        -2,
    )  # B3H'W'

    fimg = fimg.permute(0, 2, 3, 1).byte()  # BH'W'3
    img = img.permute(0, 2, 3, 1).byte()  # BH'W'3
    return {
        f'paper_{prefix}front': fimg,
        f'paper_{prefix}hmc': img,
    }


def paper_diag_input(batch, prefix=''):
    inp_img = batch['hmc_cam_img'].expand(-1, -1, 3, -1, -1)  # BN3HW
    inp_img = torch.cat(
        [
            torch.cat([inp_img[:, 0], inp_img[:, 1].flip(-1)], -1),
            torch.cat([inp_img[:, 2], inp_img[:, 3].flip(-1)], -1),
        ],
        -2,
    )  # B3H'W'
    inp_img = inp_img.permute(0, 2, 3, 1).byte()  # BH'W'3
    return {
        f'paper_{prefix}inp': inp_img,
    }


def paper_diag_style_transfer(batch, outputs, prefix=''):
    rosetta_state: State = batch['rosetta_correspondences']['state']
    mask = rosetta_state.mask  # BN1HW
    pred = outputs['pred_rgb']  # BN3HW

    # pred = color_correct_2(pred, 2)
    # pred = pred * mask + (1 - mask) * 255  # BN3HW
    pred = pred * mask
    pred = linear2color_corr(pred / 255, 2) * 255

    pred = torch.cat(
        [
            torch.cat([pred[:, 0], pred[:, 1].flip(-1)], -1),
            torch.cat([pred[:, 2], pred[:, 3].flip(-1)], -1),
        ],
        -2,
    )  # B3H'W'
    pred = pred.permute(0, 2, 3, 1).byte()  # BH'W'3
    ret = {
        f'paper_{prefix}pred': pred,
    }

    # # Conditioning images
    # cond = outputs['id_cond']  # BNM3HW
    # cond = color_correct_2(cond, 3)
    # # cond = linear2color_corr(cond / 255, 3) * 255
    # for c in range(cond.shape[2]):
    #     temp = cond[:, :, c]
    #     temp = torch.cat(
    #         [
    #             torch.cat([temp[:, 0], temp[:, 1].flip(-1)], -1),
    #             torch.cat([temp[:, 2], temp[:, 3].flip(-1)], -1),
    #         ],
    #         -2,
    #     )  # B3H'W'
    #     ret[f'paper_{prefix}cond_{c}'] = temp.permute(0, 2, 3, 1).byte()  # BH'W'3

    # Input images
    inp_img = batch['hmc_cam_img'].expand(-1, -1, 3, -1, -1)  # BN3HW
    inp_img = torch.cat(
        [
            torch.cat([inp_img[:, 0], inp_img[:, 1].flip(-1)], -1),
            torch.cat([inp_img[:, 2], inp_img[:, 3].flip(-1)], -1),
        ],
        -2,
    )  # B3H'W'
    inp_img = inp_img.permute(0, 2, 3, 1).byte()  # BH'W'3
    ret[f'paper_{prefix}inp'] = inp_img

    # GT images
    gt_img = linear2color_corr(rosetta_state.img / 255, 2) * 255
    gt_img = torch.cat(
        [
            torch.cat([gt_img[:, 0], gt_img[:, 1].flip(-1)], -1),
            torch.cat([gt_img[:, 2], gt_img[:, 3].flip(-1)], -1),
        ],
        -2,
    )  # B3H'W'
    gt_img = gt_img.permute(0, 2, 3, 1).byte()  # BH'W'3
    ret[f'paper_{prefix}gt'] = gt_img

    return ret
