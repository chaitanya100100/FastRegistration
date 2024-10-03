import torch as th


class KeypointHandler(th.nn.Module):
    def __init__(
        self,
        topology_file,
        keypoint_location_file,
        keypoint_visualize_file,
        upper_subset_file,
        lower_subset_file,
        cameras,
    ):
        super().__init__()
        self.num_views = len(cameras)

        # topology of face
        self.topology = typed.load(get_repo_asset_path(topology_file))
        vi = th.from_numpy(self.topology["vi"]).long()

        # topology - keypoint correspondence
        keypoint_location = typed.load(
            get_repo_asset_path(keypoint_location_file), extension="nptxt"
        )
        face_index = th.from_numpy(keypoint_location[:, 1]).long()
        bary_coord = th.from_numpy(keypoint_location[:, 2:4]).float()
        bary_coord = th.cat(
            (bary_coord, th.ones_like(bary_coord[:, :1]) - bary_coord.sum(-1, keepdim=True)), dim=-1
        )
        self.register_buffer("bary_coord", bary_coord, persistent=False)
        self.register_buffer("vertex_indices", vi[face_index], persistent=False)

        # keypoint visualizations
        keypoint_visualize = typed.load(get_repo_asset_path(keypoint_visualize_file))
        self.draw_point = KeypointDrawer(**keypoint_visualize)

        # --------------------
        # set up subsets
        # --------------------
        keypoint_index = th.from_numpy(keypoint_location[:, 0]).long()
        convert_index = -th.ones(int(keypoint_index.max() + 1)).long()
        convert_index[keypoint_index] = th.arange(len(keypoint_index))
        self.total_keypoints: int = len(keypoint_index)

        self.max_subset_size = 0
        subsets = []
        for subset_file in [upper_subset_file, lower_subset_file]:
            enabled_subset = typed.load(get_repo_asset_path(subset_file), extension="nptxt")
            enabled_subset = th.from_numpy(enabled_subset).long()
            enabled_subset = convert_index[enabled_subset]

            if (enabled_subset == -1).any():
                raise ValueError("Keypoint index mapping failed.")
            self.max_subset_size = max(self.max_subset_size, len(enabled_subset))
            subsets.append(enabled_subset)

        # left eye camera sees upper subset. right eye camera sees flipped upper subset.
        # both left and right camera sees lower subset.
        upper_subset, lower_subset = subsets
        upper_flipped_subset = (
            set(range(self.total_keypoints))
            - set(upper_subset.numpy().tolist())
            - set(lower_subset.numpy().tolist())
        )
        upper_flipped_subset = th.tensor(list(upper_flipped_subset)).long().sort()[0]

        subsets = []
        for c in cameras:
            if "left" in c and "eye" in c:
                subsets.append(upper_subset)
            elif "right" in c and "eye" in c:
                subsets.append(upper_flipped_subset)
            else:
                subsets.append(lower_subset)

        for i in range(self.num_views):
            self.register_buffer(f"subset_{i}", subsets[i], persistent=False)

    def get_keypoints(self, verts: th.Tensor):
        # verts: B x V x *
        return (verts[:, self.vertex_indices] * self.bary_coord[None, ..., None]).sum(2)

    def draw_keypoint(self, images, kp2d, gt_kp2d=None):
        # images: B x C x H x W
        # kp2d: B x K x 2
        images_np = images.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        # clamp keypoints to image boundary
        vis = kp2d[..., 2]
        vis = vis * (kp2d[..., 0] >= 0) * (kp2d[..., 0] < images.shape[-2])
        vis = vis * (kp2d[..., 1] >= 0) * (kp2d[..., 1] < images.shape[-1])
        kp2d[..., 2] = vis
        kp2d[..., 0].clamp_(min=0, max=images.shape[-2])
        kp2d[..., 1].clamp_(min=0, max=images.shape[-1])
        gt_kp2d = gt_kp2d if gt_kp2d is not None else [None] * images.shape[0]
        for img, kpt, gkpt in zip(images_np, kp2d, gt_kp2d):
            self.draw_point.draw(img, kpt, gt_keypoints=gkpt)
        return th.from_numpy(images_np).permute(0, 3, 1, 2).to(images.device)

    def visualize_hmc_keypoints(self, kp2d, hmc_cam_img, det_kp2d=None):
        # kp2d: B x N x K x 2
        # hmc_cam_img: B x N x C x H x W
        B, N, _, H, W = hmc_cam_img.shape
        hmc_cam_img = hmc_cam_img.view(B * N, -1, H, W)  # (BN)CHW
        if hmc_cam_img.shape[1] == 1:
            hmc_cam_img = hmc_cam_img.expand(-1, 3, -1, -1).byte()  # C->3C

        kp2d = kp2d.view(B * N, -1, 3)  # (BN)K3
        det_kp2d = det_kp2d.view(B * N, -1, 3) if det_kp2d is not None else None

        kpt_img = self.draw_keypoint(hmc_cam_img.clone(), kp2d, det_kp2d)  # (BN)CHW
        kpt_img = kpt_img.view(B, N, -1, H, W)  # BNCHW
        kpt_img = kpt_img.permute(0, 3, 1, 4, 2).reshape(B, H, N * W, -1)  # BH(NW)C
        return kpt_img

    def select_subset(self, keypoints: th.Tensor):
        # keypoints: B x N x K x X
        # output: B x N x K' x X
        # if keypoints.ndim == 3:
        #     keypoints = keypoints[:, None, ...].repeat(1, self.num_views, 1, 1)

        assert keypoints.shape[2] == self.total_keypoints
        assert keypoints.shape[1] == self.num_views
        output = []
        for i in range(self.num_views):
            subset = self.get_buffer(f"subset_{i}")
            sub_kp = keypoints[:, i, subset]
            output.append(sub_kp)

        output = pad_and_cat(output, dim=0)  # BN x K' x X
        assert output.shape[1] == self.max_subset_size
        output = output.view(-1, self.num_views, *output.shape[1:])  # B x N x K' x X
        return output

    def scatter_subset(self, keypoints: th.Tensor):
        # keypoints: B x N x K' x X
        # output: B x N x K x X
        assert keypoints.shape[1] == self.num_views
        output = th.zeros(
            keypoints.shape[0],
            keypoints.shape[1],
            self.total_keypoints,
            *keypoints.shape[3:],
            dtype=keypoints.dtype,
            device=keypoints.device,
        )

        for i in range(self.num_views):
            subset = self.get_buffer(f"subset_{i}")
            output[:, i, subset] = keypoints[:, i, : len(subset)]

        return output
