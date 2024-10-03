# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import numpy as np
import torch
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF


def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v)


def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)


def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)


def Flip(img, _):  # not from the paper
    return PIL.ImageOps.mirror(img)


def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v)


def SolarizeAdd(img, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def Posterize(img, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v)


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v)


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.0:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, v)


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, v):
    return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    al = [
        (Identity, 0.0, 1.0),
        # (AutoContrast, 0, 1),
        # (Equalize, 0, 1),
        # (Invert, 0, 1),
        # (Rotate, 0, 30),
        (Posterize, 4, 8),
        # (Solarize, 0, 256),
        # (SolarizeAdd, 0, 110),
        (Color, 0.1, 1.9),
        (Contrast, 0.1, 1.9),
        (Brightness, 0.1, 1.9),
        (Sharpness, 0.1, 1.9),
        # (ShearX, 0.0, 0.3),
        # (ShearY, 0.0, 0.3),
        # (Cutout, 0, 0.2),
        # (CutoutAbs, 0, 40),
        # (TranslateXabs, 0.0, 100),
        # (TranslateYabs, 0.0, 100),
    ]

    return al


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        if self.length <= 0:
            return img
        h, w = img.size(1), img.size(2)
        mask = torch.ones((h, w), device=img.device, dtype=img.dtype)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        img = img * mask.expand_as(img)
        return img


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m  # [0, 30]
        self.augment_list = augment_list()

    def augment(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img


def resize2d(x, size, scale_factor=None):
    if size is None:
        assert scale_factor is not None
        size = tuple(int(s * scale_factor) for s in x.shape[-2:])
    size = tuple(size)
    assert len(size) == 2
    assert x.dim() >= 4

    bdim = x.shape[:-3]
    x = x.flatten(0, -4)
    x = torch.nn.functional.interpolate(x, size, mode="bilinear", align_corners=False)
    x = x.unflatten(0, bdim)
    return x


class MyAugment:
    def __init__(self, scale) -> None:
        assert 0 <= scale <= 10
        scale = scale / 10
        self.scale = scale
        self.jitter = torchvision.transforms.ColorJitter(
            brightness=scale / 4, contrast=scale / 4, saturation=scale / 4, hue=scale / 8
        )
        self.blur = torchvision.transforms.GaussianBlur(kernel_size=11, sigma=(0.2, scale * 5))

    def augment_one_img(self, img, only_geo=False):
        # apply same augmentation on the all views of both image stacks
        # img: B3HW
        _, _, H, _ = img.shape
        img = img / 255

        # Photometric augmentations
        if not only_geo:
            ret = []
            for im in img:
                if random.random() < 0.5:
                    im = self.jitter(TF.to_pil_image(im))
                    im = TF.to_tensor(im).to(img.device)
                ret.append(im)
            ret = torch.stack(ret)
        else:
            ret = img

        # Cutout and Blur
        ret2 = []
        for im in ret:
            if random.random() < 0.5:
                le = min(H, int(2 * self.scale * H))
                le = random.randint(le // 4, le)
                im = CutoutDefault(le)(im)
            if random.random() < 0.5:
                im = self.blur(im)
            ret2.append(im)

        ret2 = torch.stack(ret2) * 255
        return ret2

    def augment_hmc(self, img1, img2):
        # apply same augmentation on the all views of both image stacks
        # img1, img2: BN3HW
        B, N, _, H, _ = img1.shape
        img = torch.cat([img1, img2], dim=1) / 255  # B(2N)3HW
        img = img.permute(0, 2, 1, 3, 4).flatten(2, 3)  # B3(2NH)W

        # Photometric augmentations
        ret = []
        for im in img:
            if random.random() < 0.5:
                im = self.jitter(TF.to_pil_image(im))
                im = TF.to_tensor(im).to(img1.device)
            ret.append(im)
        ret = torch.stack(ret)

        ret = ret.unflatten(2, (2 * N, H)).permute(0, 2, 1, 3, 4)  # B(2N)3HW
        ret = ret.flatten(0, 1)  # (B2N)3HW

        # Cutout and Blur
        ret2 = []
        for im in ret:
            if random.random() < 0.5:
                le = min(H, int(2 * self.scale * H))
                le = random.randint(le // 4, le)
                im = CutoutDefault(le)(im)
            if random.random() < 0.5:
                im = self.blur(im)
            ret2.append(im)

        ret2 = torch.stack(ret2) * 255

        img1, img2 = ret2.unflatten(0, (B, 2 * N)).split(N, dim=1)
        return img1.contiguous(), img2.contiguous()

    def augment_st(self, hmc_img, gt_img, cond_img, gt_mask, cond_mask, cond_uv):
        # hmc_img: BN1HW
        # gt_img: BN3HW
        # cond_img: BNM3HW
        hmc_img = hmc_img.expand(-1, -1, 3, -1, -1)
        B, N, M, _, H, W = cond_img.shape
        device = hmc_img.device

        def _augment(im, x, y, h, w):
            im = im[..., x : x + h, y : y + w]
            im = resize2d(im, (H, W))
            return im

        new_hmc_img = []
        new_gt_img = []
        new_cond_img = []
        new_gt_mask = []
        new_cond_mask = []
        new_cond_uv = []

        min_scale = max(0.1 / self.scale, 0.2)
        for i in range(B):
            x, y, h, w = torchvision.transforms.RandomResizedCrop.get_params(
                hmc_img[i, 0], (min_scale, 1.0), (3.0 / 4.0, 4.0 / 3.0)
            )
            hi = _augment(hmc_img[i], x, y, h, w)
            gi = _augment(gt_img[i], x, y, h, w)
            ci = _augment(cond_img[i], x, y, h, w)
            gm = _augment(gt_mask[i], x, y, h, w)
            cm = _augment(cond_mask[i], x, y, h, w)
            cu = _augment(cond_uv[i], x, y, h, w)

            if random.random() < 0.5:
                combined = torch.cat([hi, gi, ci.flatten(0, 1)], dim=0)  # (N+N+NM)3HW
                combined = combined.permute(1, 0, 2, 3).flatten(1, 2)  # 3((N+N+NM)H)W
                combined = self.jitter(TF.to_pil_image(combined / 255))
                combined = TF.to_tensor(combined).to(device) * 255
                combined = combined.unflatten(1, (N + N + N * M, H)).transpose(0, 1)  # (N+N+NM)3HW
                hi, gi, ci = combined.split([N, N, N * M], dim=0)
                ci = ci.unflatten(0, (N, M))

            new_hmc_img.append(hi)
            new_gt_img.append(gi)
            new_cond_img.append(ci)
            new_gt_mask.append(gm)
            new_cond_mask.append(cm)
            new_cond_uv.append(cu)

        new_hmc_img = torch.stack(new_hmc_img)[:, :, :1]
        new_gt_img = torch.stack(new_gt_img)
        new_cond_img = torch.stack(new_cond_img)
        new_gt_mask = torch.stack(new_gt_mask)
        new_cond_mask = torch.stack(new_cond_mask)
        new_cond_uv = torch.stack(new_cond_uv)

        return new_hmc_img, new_gt_img, new_cond_img, new_gt_mask, new_cond_mask, new_cond_uv


# class MyAugment2:
#     def __init__(self, scale) -> None:
#         assert 0 <= scale <= 10
#         scale = scale / 10

#         import sys

#         sys.path.append("/mnt/home/chpatel/packages/mytemp")
#         import albumentations as A
#         import torchvision.transforms as T

#         print(scale)

#         additional = {f"image{i}": "image" for i in range(16)}
#         additional.update({f"mask{i}": "mask" for i in range(16)})
#         self.suff = [''] + [f"{i}" for i in range(16)]

#         brmax = (int(scale * 32) // 2) * 2 + 1
#         self.pixel_aug = A.Compose(
#             [
#                 A.ColorJitter(
#                     brightness=scale / 5,
#                     contrast=scale / 5,
#                     saturation=scale / 5,
#                     hue=scale / 10,
#                     p=0.5,
#                 ),
#                 A.CoarseDropout(max_holes=int(16 * scale), max_height=32, max_width=32, p=0.5),
#                 A.GaussianBlur(blur_limit=(3, brmax), sigma_limit=(0.5, 5 * scale), p=0.5),
#             ],
#             additional_targets=additional,
#         )

#     def augment_hmc(self, img1, img2):
#         # apply same augmentation on the all views of both image stacks
#         # img1, img2: BN3HW
#         B, N = img1.shape[:2]
#         img = torch.stack([img1, img2], dim=2).flatten(0, 1).byte()  # (BN)23HW
#         # img = torch.stack([img1, img2], dim=2).flatten(1, 2) / 255  # B(N2)3HW

#         # bn3hw
#         # b = img.shape[0]
#         n = img.shape[1]
#         img = img.permute(0, 1, 3, 4, 2).cpu().numpy()  # bnHW3

#         # Photometric augmentations
#         ret = []
#         suff = self.suff
#         for im in img:
#             im = {f"image{suff[i]}": im[i] for i in range(n)}
#             im = self.pixel_aug(**im)
#             im = [255 * TF.to_tensor(im[f"image{suff[i]}"]) for i in range(n)]
#             im = torch.stack(im)  # n3HW
#             ret.append(im)
#         ret = torch.stack(ret).float().to(img1.device)  # bn3HW

#         ret = ret.unflatten(0, (B, N))  # BN23HW
#         # ret = ret.unflatten(1, (N, 2))  # BN23HW

#         img1, img2 = ret[:, :, 0], ret[:, :, 1]
#         return img1.contiguous(), img2.contiguous()
