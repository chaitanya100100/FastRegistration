import random
import torch
import numpy as np
import cv2
from pytorch3d.transforms.rotation_conversions import rotation_6d_to_matrix, matrix_to_rotation_6d


def get_error(a, b, weight=None, loss_type='l2', sum_dim=None, center_wrt=None, mean_dim=None):
    # a, b: B x N x *
    if center_wrt is not None:
        a = a - a[:, center_wrt : center_wrt + 1]
        b = b - b[:, center_wrt : center_wrt + 1]

    diff = a - b

    if loss_type == 'l1':
        loss = diff.abs()
    elif loss_type == 'l2':
        loss = diff.square()
    elif loss_type == 'residual':
        loss = diff
    elif loss_type.startswith('huber'):
        delta = float(loss_type.split('_')[1])
        loss = torch.nn.functional.huber_loss(a, b, delta=delta, reduction='none')
    else:
        raise AttributeError

    if weight is not None:
        loss = loss * weight

    assert sum_dim is None or mean_dim is None
    if sum_dim is not None:
        loss = loss.sum(sum_dim)
    if mean_dim is not None:
        loss = loss.mean(mean_dim)

    return loss


def jac_reproj_error_wrt_kp2d(pred_kp2d, tar_kp2d, confidence, error_type='l1', center_wrt=None):
    """Returns Jacobian of reprojection error vector w.r.t. predicted 2d keypoints. Works with batched input.

    Note that if center_wrt is not None, then the jacobian of each error_i will depend on that kp_i and center kp_c. This
    function returns jacobian of error_i wrt that kp_i only. Jacobian wrt center kp_c will be negative of that.

    Args:
        pred_kp2d, tar_kp2d: N x V x 2
        confidence: N x V
        error_type: Either l1 or l2 loss or residual.
        center_wrt: If not None, it should be an index of root center. The loss will be computed wrt that
            root joint center.
    Returns:
        jac: N x V x 2
    """
    assert error_type in ['l1', 'l2', 'residual']
    if center_wrt is not None:
        pred_kp2d = pred_kp2d - pred_kp2d[:, center_wrt : center_wrt + 1]
        tar_kp2d = tar_kp2d - tar_kp2d[:, center_wrt : center_wrt + 1]

    if error_type == 'l1':
        jac = torch.sign(pred_kp2d - tar_kp2d)  # N x V x 2
    elif error_type == 'l2':
        jac = 2 * (pred_kp2d - tar_kp2d)  # N x V x 2
    else:
        jac = torch.ones_like(pred_kp2d)
    jac = jac * confidence[..., None]  # N x V x 2

    return jac


def jac_kp2d_wrt_kp3d(v, campos, camrot, focal, princpt):
    """Returns Jacobian of projected kp2d w.r.t. kp3d. It involves transformation to camera coordinate and projection.
    Args:
        v:                  N x V x 3
        camrot:             N x 3 x 3
        campos:             N x 3
        focal:              N x 2 x 2
        princpt:            N x 2

    Returns:
        jac_vpix_v: N x V x 2 x 3
    """
    cam_frame = campos is None and camrot is None
    if not cam_frame:
        v_cam = (camrot[:, None] @ (v - campos[:, None])[..., None])[..., 0]  # N x V x 3
    else:
        v_cam = v

    z = v_cam[:, :, 2:3]
    z = torch.where(z < 0, z.clamp(max=-1e-8), z.clamp(min=1e-8))  # N x V x 1

    temp1 = focal[:, None] / z[..., None]  # N x V x 2 x 2
    temp2 = -(focal[:, None] @ v_cam[..., :2, None]) / (z[..., None] ** 2)  # N x V x 2 x 1
    jac_vpix_vcam = torch.cat([temp1, temp2], -1)  # N x V x 2 x 3

    if not cam_frame:
        jac_vcam_v = camrot[:, None]  # N x 1 x 3 x 3
        jac_vpix_v = jac_vpix_vcam @ jac_vcam_v  # N x V x 2 x 3
    else:
        jac_vpix_v = jac_vpix_vcam

    return jac_vpix_v


def jac_kp2d_wrt_campos_camrot(v, campos, camrot, focal, princpt):
    """Returns Jacobian of projected kp2d w.r.t. campos and camrot.
    Args:
        v:                  N x V x 3
        camrot:             N x 3 x 3
        campos:             N x 3
        focal:              N x 2 x 2
        princpt:            N x 2

    Returns:
        jac_vpix_campos: N x V x 2 x 3
        jac_vpix_camrot: N x V x 2 x 3 x 3
    """
    v_cam = (camrot[:, None] @ (v - campos[:, None])[..., None])[..., 0]  # N x V x 3
    jac_vpix_vcam = jac_kp2d_wrt_kp3d(v_cam, None, None, focal, princpt)  # N x V x 2 x 3

    # jac_vcam_camrot  N x V x 3 x 3 x 3 (nvijk) where i==j row has same vector as N x V x 3
    # jac_vcam_campos  N x V x 3 x 3 (nvij) where every v index has the same matrix  N x 3 x 3
    jac_vcam_camrot = v - campos[:, None]
    jac_vcam_campos = -camrot

    # jac_vpix_camrot N x V x 2 x 3 x 3
    # jac_vpix_campos N x V x 2 x 3
    jac_vpix_camrot = torch.einsum('nvij,nvk->nvijk', jac_vpix_vcam, jac_vcam_camrot)
    jac_vpix_campos = torch.einsum('nvij,njk->nvik', jac_vpix_vcam, jac_vcam_campos)
    return jac_vpix_campos, jac_vpix_camrot


def aa_to_rotmat_direct(k, t=None):
    """Computes the rotation matrix R from a tensor of Rodrigues vectors (also known as axis angles).

    This is yet another implementation of same Rodrigues formula. This helped to define jacobian easily.
    Input is either axis angle vector `aa` or `(k, t)` where `k` is a unit vector defining rotation axis
    and `t` defines magnitude of rotation.

    Args:
        k: ... x 3
        t: ... (optional)
    Returns:
        rotmat: ... x 3 x 3
    """
    if t is None:
        t = torch.norm(k, p=2, dim=-1)
        k = k / (t[..., None] + 1.0e-9)

    cost = torch.cos(t)
    sint = torch.sin(t)
    kx, ky, kz = [x[..., 0] for x in torch.split(k, 1, dim=-1)]

    rotmat = torch.stack(
        [
            torch.stack(
                [
                    cost + kx**2 * (1 - cost),
                    kx * ky * (1 - cost) - kz * sint,
                    kx * kz * (1 - cost) + ky * sint,
                ],
                -1,
            ),
            torch.stack(
                [
                    kx * ky * (1 - cost) + kz * sint,
                    cost + ky**2 * (1 - cost),
                    ky * kz * (1 - cost) - kx * sint,
                ],
                -1,
            ),
            torch.stack(
                [
                    kx * kz * (1 - cost) - ky * sint,
                    ky * kz * (1 - cost) + kx * sint,
                    cost + kz**2 * (1 - cost),
                ],
                -1,
            ),
        ],
        -2,
    )
    return rotmat


def aa_to_rotmat(aa):
    return aa_to_rotmat_direct(aa)


def rotmat_to_rot6d(rotmat):
    return matrix_to_rotation_6d(rotmat)


def rot6d_to_rotmat(rot6d):
    return rotation_6d_to_matrix(rot6d)


def skew_symmetric_matrix(a):
    # a: ... x 3
    # returns ... x 3 x 3
    a1, a2, a3 = a.unbind(-1)
    z0 = torch.zeros_like(a1)
    c1 = torch.stack([z0, a3, -a2], -1)
    c2 = torch.stack([-a3, z0, a1], -1)
    c3 = torch.stack([a2, -a1, z0], -1)
    ret = torch.stack([c1, c2, c3], -1)
    return ret


def jac_rotmat_wrt_rot6d(rot6d):
    # rot6d: ... x 6 refered to as B x 6
    # returns ... x 3 x 3 x 6
    batch_dim = rot6d.size()[:-1]
    e3 = torch.eye(3, device=rot6d.device).repeat(batch_dim + (1, 1))  # B x 3 x 3
    z0 = torch.zeros_like(e3)  # B x 3 x 3
    a1, a2 = rot6d[..., :3], rot6d[..., 3:]  # B x 3

    # b1 = F.normalize(a1, dim=-1)
    b1 = torch.nn.functional.normalize(a1, dim=-1)  # B x 3
    jac_b1_a1 = jac_normalized_vector_wrt_vector(a1)  # B x 3 x 3
    jac_b1_a2 = z0  # B x 3 x 3

    # b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    temp = (b1 * a2).sum(-1)  # B
    jac_b2_b1 = (-b1[..., None] * a2[..., None, :]) - e3 * temp[..., None, None]  # B x 3 x 3
    jac_b2_a1 = torch.matmul(jac_b2_b1, jac_b1_a1)

    jac_b2_a2 = e3 - b1[..., None] * b1[..., None, :]  # B x 3 x 3

    # b2 = F.normalize(b2, dim=-1)
    jac_b2_oldb2 = jac_normalized_vector_wrt_vector(b2)  # B x 3 x 3
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    jac_b2_a1 = torch.matmul(jac_b2_oldb2, jac_b2_a1)
    jac_b2_a2 = torch.matmul(jac_b2_oldb2, jac_b2_a2)

    # b3 = torch.cross(b1, b2, dim=-1)
    jac_b3_b1 = -skew_symmetric_matrix(b2)  # B x 3 x 3
    jac_b3_b2 = skew_symmetric_matrix(b1)  # B x 3 x 3
    jac_b3_a1 = torch.matmul(jac_b3_b1, jac_b1_a1) + torch.matmul(jac_b3_b2, jac_b2_a1)  # B x 3 x 3
    jac_b3_a2 = torch.matmul(jac_b3_b1, jac_b1_a2) + torch.matmul(jac_b3_b2, jac_b2_a2)  # B x 3 x 3

    jac_a1 = torch.stack([jac_b1_a1, jac_b2_a1, jac_b3_a1], -3)  # B x 3 x 3 x 3
    jac_a2 = torch.stack([jac_b1_a2, jac_b2_a2, jac_b3_a2], -3)  # B x 3 x 3 x 3
    return torch.cat([jac_a1, jac_a2], -1)  # B x 3 x 3 x 6


def jac_normalized_vector_wrt_vector(x):
    # jacobian of normalized vector wrt original vector
    # x: ... x 3
    # returns ... x 3 x 3
    batch_dim = x.shape[:-1]
    xnorm = torch.norm(x, p=2, dim=-1)
    v = x / xnorm[..., None]
    e = torch.eye(3, device=x.device).repeat(batch_dim + (1, 1))
    jac = (e - v[..., None] * v[..., None, :]) / xnorm[..., None, None]
    return jac


def jac_norm_wrt_vector(x):
    # jacobian of vector norm wrt vector
    # x: ... x 3
    # returns ... x 3
    xnorm = torch.norm(x, p=2, dim=-1)
    v = x / xnorm[..., None]
    return v


def jac_rotmat_wrt_axis_and_angle(k, t):
    """Jacobian of rotation matrix wrt unit rotation axis `k` and rotation angle `t`.

    This function expects unit-length axis `k` and the magnitude of rotation `t`. This is a helper
    function for computing jacobian of axis_angle_to_rotmat.

    Args:
        k: ... x 3
        t: ... (optional)
    Returns:
        jac_k: ... x 3 x 3 x 3
        jac_t: ... x 3 x 3
    """
    cost = torch.cos(t)
    sint = torch.sin(t)
    kx, ky, kz = [x[..., 0] for x in torch.split(k, 1, dim=-1)]

    jac_t = torch.stack(
        [
            torch.stack(
                [
                    -sint + kx**2 * sint,
                    kx * ky * sint - kz * cost,
                    kx * kz * sint + ky * cost,
                ],
                -1,
            ),
            torch.stack(
                [
                    kx * ky * sint + kz * cost,
                    -sint + ky**2 * sint,
                    ky * kz * sint - kx * cost,
                ],
                -1,
            ),
            torch.stack(
                [
                    kx * kz * sint - ky * cost,
                    ky * kz * sint + kx * cost,
                    -sint + kz**2 * sint,
                ],
                -1,
            ),
        ],
        -2,
    )

    # * x 3 x 3 x 3
    zero = torch.zeros_like(t)
    jac_kx = torch.stack(
        [
            torch.stack([2 * kx * (1 - cost), ky * (1 - cost), kz * (1 - cost)], -1),
            torch.stack([ky * (1 - cost), zero, -sint], -1),
            torch.stack([kz * (1 - cost), sint, zero], -1),
        ],
        -2,
    )
    jac_ky = torch.stack(
        [
            torch.stack([zero, kx * (1 - cost), sint], -1),
            torch.stack([kx * (1 - cost), 2 * ky * (1 - cost), kz * (1 - cost)], -1),
            torch.stack([-sint, kz * (1 - cost), zero], -1),
        ],
        -2,
    )
    jac_kz = torch.stack(
        [
            torch.stack([zero, -sint, kx * (1 - cost)], -1),
            torch.stack([+sint, zero, ky * (1 - cost)], -1),
            torch.stack([kx * (1 - cost), ky * (1 - cost), 2 * kz * (1 - cost)], -1),
        ],
        -2,
    )
    jac_k = torch.stack([jac_kx, jac_ky, jac_kz], -1)  # * x 3 x 3 x 3

    return jac_k, jac_t


def jac_rotmat_wrt_axis_angle(aa):
    """Jacobian of rotation matrix wrt axis angle `aa`.

    `aa` defines rotation by Rodrigues formula - also known as axis angle or rot vecs.

    Args:
        aa: ... x 3
    Returns:
        jac_mat_aa: ... x 3 x 3 x 3
    """
    t = torch.norm(aa, p=2, dim=-1)
    k = aa / (t[..., None] + 1.0e-9)
    # * x 3 x 3 x 3 and * x 3 x 3
    jac_mat_k, jac_mat_t = jac_rotmat_wrt_axis_and_angle(k, t)

    jac_k_aa = jac_normalized_vector_wrt_vector(aa)  # * x 3 x 3
    jac_t_aa = jac_norm_wrt_vector(aa)  # * x 3

    jac_mat_aa = torch.einsum('...ijk,...kl->...ijl', jac_mat_k, jac_k_aa) + torch.einsum(
        '...ij,...l->...ijl', jac_mat_t, jac_t_aa
    )
    return jac_mat_aa


def jac_camrot_campos_wrt_a2h_R_t(hmc_Rt, a2h_R, a2h_t):
    """Jacobian of camrot and campos wrt avatar2hmc_R and avatar2hmc_t.

    Points in avatar space are transformed into HMC space using a2h_R and a2h_t
    which are transformed into HMC camera space using hmc_Rt.

    Args:
        hmc_Rt: B x 4 x 4 denoting the transformation of HMC cameras in HMC space.
        a2h_R, a2h_t: B x 3 x 3 and B x 3 denoting the transformation from avatar
            space to HMC space.
    Returns:
        jac_cR_Ra: B x 3 x 3 x 3 x 3
        jac_cR_ta: B x 3 x 3 x 3
        jac_cp_Ra: B x 3 x 3 x 3
        jac_cp_ta: B x 3 x 3

        where cR=camrot, cp=campos, Ra=a2h_R, ta=a2h_t
    """
    B = hmc_Rt.shape[0]
    hmc_R, hmc_t = hmc_Rt[..., :3, :3], hmc_Rt[..., :3, 3]

    jac_cR_Ra = hmc_R[:, :, :, None].expand(-1, -1, -1, 3).diag_embed().permute(0, 1, 4, 2, 3)
    jac_cR_ta = torch.zeros(B, 3, 3, 3, device=hmc_Rt.device, dtype=hmc_Rt.dtype)

    x = -a2h_t - (hmc_R.transpose(-2, -1) @ hmc_t[..., None])[..., 0]
    jac_cp_Ra = x[..., None].expand(-1, -1, 3).diag_embed().permute(0, 2, 1, 3)

    jac_cp_ta = -a2h_R.transpose(-2, -1)

    # B3333, B333, B333, B33
    return jac_cR_Ra, jac_cR_ta, jac_cp_Ra, jac_cp_ta


def jac_camrotpos_wrt_rtvec(hmc_Rt, rtvec):
    """Jacobian of camrot and campos wrt avatar2hmc rvec and tvec.

    Fwd pass is rtvec -> a2h_Rt(axis angle to matrix) -> Rt (by multiplying with hmc_Rt)
    -> camrot, campos (R, -R^Tt). This method outputs jacobian for this fwd pass.
    This method can be optimized further.

    Edit: Now allowing 6d rotation as well. Tensor size comments are still for 3d rotation.

    Args:
        hmc_Rt: B x N x 3 x 4 extrinsics of each HMC camera in HMC space
        rtvec: B x 6 rotation and translation of avatar in HMC space.
            rtvec[:, :3] is rotation (axis-angle) and rtvec[:, 3:] is translation.
    Returns:
        jac_camrotpos_rtvec: (BN) x 3 x 4 x 6
    """
    B, N = hmc_Rt.shape[:2]

    hmc_Rt = pad_3x4_to_4x4(hmc_Rt).view(B * N, 4, 4)  # (BN)44

    if rtvec.shape[-1] == 6:
        rvec = rtvec[..., :3][:, None].expand(-1, N, -1).flatten(0, 1)  # B3 -> (BN)3
        tvec = rtvec[..., 3:][:, None].expand(-1, N, -1).flatten(0, 1)  # B3 -> (BN)3
        a2h_R, a2h_t = rvec_to_R(rvec), tvec  # (BN)33, (BN)3
    elif rtvec.shape[-1] == 9:
        rvec = rtvec[..., :6][:, None].expand(-1, N, -1).flatten(0, 1)  # B6 -> (BN)6
        tvec = rtvec[..., 6:][:, None].expand(-1, N, -1).flatten(0, 1)  # B3 -> (BN)3
        a2h_R, a2h_t = rot6d_to_rotmat(rvec), tvec  # (BN)33, (BN)3
    else:
        raise NotImplementedError

    # (BN)3333, (BN)333, (BN)333, (BN)33
    (
        jac_camrot_a2hR,
        jac_camrot_tvec,
        jac_campos_a2hR,
        jac_campos_tvec,
    ) = jac_camrot_campos_wrt_a2h_R_t(hmc_Rt, a2h_R, a2h_t)
    if rtvec.shape[-1] == 6:
        jac_a2hR_rvec = jac_rotmat_wrt_axis_angle(rvec)  # (BN)333
    elif rtvec.shape[-1] == 9:
        jac_a2hR_rvec = jac_rotmat_wrt_rot6d(rvec)  # (BN)336
    else:
        raise NotImplementedError

    jac_camrot_rvec = torch.einsum('Bijkl,Bklm->Bijm', jac_camrot_a2hR, jac_a2hR_rvec)  # (BN)333
    jac_campos_rvec = torch.einsum('Bikl,Bklm->Bim', jac_campos_a2hR, jac_a2hR_rvec)  # (BN)33

    # Concate all things
    # (BN)343 and (BN)343
    jac_camrotpos_rvec = torch.cat([jac_camrot_rvec, jac_campos_rvec[:, :, None, :]], -2)
    jac_camrotpos_tvec = torch.cat([jac_camrot_tvec, jac_campos_tvec[:, :, None, :]], -2)
    # (BN)346
    jac_camrotpos_rtvec = torch.cat([jac_camrotpos_rvec, jac_camrotpos_tvec], -1)
    return jac_camrotpos_rtvec


def undistort_points(
    detected_p2ds: np.ndarray, dist_coeffs: np.ndarray, intrinsics: np.ndarray, model: str
) -> np.ndarray:
    input_coordinates: np.ndarray = detected_p2ds[:, :2].reshape(1, -1, 2)

    if model == "fisheye":
        undistortedPoints = cv2.fisheye.undistortPoints(  # pylint: disable=no-member
            input_coordinates, intrinsics, dist_coeffs
        )
    elif model == "radial-tangential":
        undistortedPoints = cv2.undistortPoints(input_coordinates, intrinsics, dist_coeffs)
    else:
        assert 0, "Unsupported camera model!!"

    p2ds: np.ndarray = undistortedPoints.reshape(-1, 2).T
    p2ds_: np.ndarray = np.vstack((p2ds, np.ones((p2ds.shape[1]), dtype=p2ds.dtype)))
    undistorted_p2ds: np.ndarray = intrinsics.dot(p2ds_)

    return undistorted_p2ds.T[:, :2]


def add_front_camera(camera_params, B: int, N: int, device):
    front_cam = frontal_view_camera_params(B, device)
    ret = {}
    for k, v in camera_params.items():
        vv = front_cam[k]
        if isinstance(v, torch.Tensor):
            v = torch.cat([v.unflatten(0, (B, N)), vv[:, None]], 1).flatten(0, 1)
        elif isinstance(v, list):
            assert len(v) == B * N
            for i in range(B)[::-1]:
                v.insert(N * (i + 1), vv[i])
        elif isinstance(v, (int, float)):
            pass
        else:
            raise NotImplementedError
        ret[k] = v
    return ret


def remove_front_camera(camera_params, B: int, N: int):
    ret = {}
    for k, v in camera_params.items():
        if isinstance(v, torch.Tensor):
            v = v.unflatten(0, (B, N))[:, :-1].flatten(0, 1)
        elif isinstance(v, list):
            assert len(v) == B * N
            for i in range(B)[::-1]:
                del v[N * i + N - 1]
        elif isinstance(v, (int, float)):
            pass
        else:
            raise NotImplementedError
        ret[k] = v
    return ret


def gather_camera_params(
    rtvec, hmc_Rt, hmc_K, render_size, hmc_distmodel=None, hmc_distcoeffs=None, isMouthView=None
):
    """Gather camera parameters to render decoder mesh from HMC viewpoints.

    Edit: Now allowing 6d rotation as well. Tensor size comments are still for 3d rotation.

    Args:
        rtvec: B x 6 where rtvec[:,:3] and rtvec[:,3:] denotes rotation (axis-angle) and
            translation for avatar2hmc transform (avatar frame to HMC frame).
        hmc_Rt: B x N x 3 x 4 denoting extrinsics of HMC cameras in HMC frame.
        hmc_K: B x N x 3 x 3 denoting intrinsics of HMC cameras.
        render_size: (int, int) denoting render size
        hmc_distmodel: Optional B x N list of strings denoting distortion model.
        hmc_distcoeffs: Optional B x N x 4 denoting distortion coefficients.
        isMouthView: Optional N list of bools denoting whether it is a mouth view.
    Returns:
        A dict containing the follwing params directly passable to functions like
        project_points, render_mvp, etc. to operate on decoder geometry.

        camrot: BN x 3 x 3
        campos: BN x 3
        focal: BN x 2 x 2
        princpt: BN x 2
        distortion_mode: BN list of strings
        distortion_coeffs: BN x 4
        isMouthView: BN list of bools
    """
    B, N = hmc_Rt.shape[:2]

    h_Rt = pad_3x4_to_4x4(hmc_Rt).view(B * N, 4, 4)  # (BN)44

    if rtvec.shape[-1] == 6:
        rvec = rtvec[..., :3][:, None].expand(-1, N, -1).flatten(0, 1)  # B3 -> (BN)3
        tvec = rtvec[..., 3:][:, None].expand(-1, N, -1).flatten(0, 1)  # B3 -> (BN)3
        a2h_R, a2h_t = rvec_to_R(rvec), tvec  # (BN)33, (BN)3
    elif rtvec.shape[-1] == 9:
        rvec = rtvec[..., :6][:, None].expand(-1, N, -1).flatten(0, 1)  # B6 -> (BN)6
        tvec = rtvec[..., 6:][:, None].expand(-1, N, -1).flatten(0, 1)  # B3 -> (BN)3
        a2h_R, a2h_t = rot6d_to_rotmat(rvec), tvec  # (BN)33, (BN)3
    else:
        raise NotImplementedError
    a2h_Rt = pad_3x4_to_4x4(torch.cat([a2h_R, a2h_t[..., None]], -1))  # (BN)44

    Rt = h_Rt @ a2h_Rt  # (BN)44

    K = hmc_K.view(-1, 3, 3)
    camera_params = {
        # "Rt": Rt,
        # "K": K,
        "camrot": Rt[:, :3, :3],
        "campos": -(Rt[:, :3, :3].transpose(-2, -1) @ Rt[:, :3, 3:])[..., 0],
        "focal": K[:, :2, :2],
        "princpt": K[:, :2, 2],
    }
    if render_size is not None:
        assert isMouthView is not None
        H, W = 400, 400  # default render size
        h, w = render_size if render_size is not None else (H, W)
        assert h == w, "I did it for simplicity. Change below if needed."
        scale = h / H
        camera_params['princpt'] = camera_params['princpt'] * scale
        camera_params['focal'] = camera_params['focal'] * scale
        camera_params['render_h'] = h
        camera_params['render_w'] = w
    if hmc_distmodel is not None:
        assert hmc_distcoeffs is not None
        camera_params.update(
            {
                "distortion_mode": [
                    s for cam_distmode in zip(*hmc_distmodel) for s in cam_distmode
                ],
                "distortion_coeff": hmc_distcoeffs.view(-1, 4),
            }
        )
    if isMouthView is not None:
        assert len(isMouthView) == N
        isMouthView_stacked = [s for _ in range(B) for s in isMouthView]
        camera_params.update({"isMouthView": isMouthView_stacked})
    return camera_params


def undistort_hmc_keypoints_batch(
    hmc_keypoints, hmc_K, hmc_distmodel, hmc_distcoeffs, img_size=None
):
    """A wrapper for undistorting keypoints for a batch of samples - each with multiple views.

    Args:
        hmc_keypoints: B x N x K x 3
        hmc_K: B x N x 3 x 3
        hmc_distmodel: B x N list of string
        hmc_distcoeffs: B x N x 4
        img_size: optional pair of ints denoting image size. useful to clamp output.
    Returns:
        hmc_keypoints_undist: B x N x K x 3
    """
    B, N = hmc_K.shape[:2]
    K = hmc_K.view(B * N, 3, 3).cpu().numpy()
    keypoints = hmc_keypoints.view(B * N, -1, 3).cpu().numpy()[..., :2]
    dist_mode = [s for cam_distmode in zip(*hmc_distmodel) for s in cam_distmode]
    dist_coeffs = hmc_distcoeffs.view(B * N, 4).cpu().numpy()
    keypoints_undist = []
    for i in range(B * N):
        udkp = undistort_points(keypoints[i], dist_coeffs[i], K[i], dist_mode[i])
        keypoints_undist.append(udkp)
    keypoints_undist = torch.from_numpy(np.stack(keypoints_undist)).to(hmc_K.dtype).to(hmc_K.device)
    keypoints_undist = torch.cat([keypoints_undist.view(B, N, -1, 2), hmc_keypoints[..., 2:]], -1)
    if img_size is not None:
        dist0 = torch.maximum(-keypoints_undist[..., 0], keypoints_undist[..., 0] - img_size[0])
        dist1 = torch.maximum(-keypoints_undist[..., 1], keypoints_undist[..., 1] - img_size[1])
        dist = torch.maximum(dist0, dist1)
        keypoints_undist[..., 2] = torch.where(
            dist > 0, torch.zeros_like(dist), keypoints_undist[..., 2]
        )
        keypoints_undist[..., 0].clamp_(min=0, max=img_size[0])
        keypoints_undist[..., 1].clamp_(min=0, max=img_size[1])
    return keypoints_undist


def undistort_image(image, K, P, D):
    if P.lower() == "fisheye":
        # pylint: disable=no-member
        image = cv2.fisheye.undistortImage(image, K, D[..., :4], Knew=K)
    else:
        image = cv2.undistort(image, K, D[..., :5])
    return image


def undistort_hmc_cam_img_batch(hmc_cam_img, hmc_K, hmc_distmodel, hmc_distcoeffs):
    B, N = hmc_cam_img.shape[:2]
    img = hmc_cam_img.flatten(0, 1).permute(0, 2, 3, 1).cpu().numpy()  # BNCHW -> (BN)HWC
    dist_mode = [s for cam_distmode in zip(*hmc_distmodel) for s in cam_distmode]
    dist_coeffs = hmc_distcoeffs.view(B * N, 4).cpu().numpy()
    K = hmc_K.view(B * N, 3, 3).cpu().numpy()

    assert len(dist_mode) == B * N

    ret = []
    for i in range(B * N):
        imud = undistort_image(img[i], K[i], dist_mode[i], dist_coeffs[i])
        ret.append(imud[..., None])

    ret = torch.from_numpy(np.stack(ret)).to(hmc_cam_img.dtype).to(hmc_cam_img.device)
    ret = ret.permute(0, 3, 1, 2).unflatten(0, (B, N))
    return ret


def frontal_view_camera_params(batch_size, device, scale=1.0):
    dd = {'device': device, 'dtype': torch.float}
    render_h, render_w = 2048, 1334
    campos = torch.tensor([[115.3817, 26.1830, 984.4196]], **dd)
    camrot = torch.tensor(
        [
            [
                [0.9913, 0.0173, -0.1303],
                [0.0244, -0.9983, 0.0531],
                [-0.1291, -0.0558, -0.9901],
            ]
        ],
        **dd,
    )
    focal = torch.tensor([[[5068.6011, 0.0000], [0.0000, 5063.9287]]], **dd)
    princpt = torch.tensor([[751.2227, 967.2575]], **dd)
    distortion_mode = "fisheye"
    distortion_coeff = torch.tensor([[0.0, 0.0, 0.0, 0.0]], **dd)
    camera_params = {
        "camrot": camrot.expand(batch_size, -1, -1),
        "campos": campos.expand(batch_size, -1),
        "focal": focal.expand(batch_size, -1, -1) * scale,
        "princpt": princpt.expand(batch_size, -1) * scale,
        "distortion_mode": [distortion_mode] * batch_size,
        "distortion_coeff": distortion_coeff.expand(batch_size, -1),
        "render_h": int(render_h * scale),
        "render_w": int(render_w * scale),
    }
    return camera_params


def put_text_on_img(img_inp, text, font_scale=3):
    # Handle torch.Tensor input
    if isinstance(img_inp, torch.Tensor):
        img = img_inp.detach().cpu().numpy()
    else:
        img = img_inp

    # Handle batched inputs
    if len(img_inp.shape) == 4:
        img = np.stack([put_text_on_img(x, text, font_scale) for x in img])
        if isinstance(img_inp, torch.Tensor):
            img = torch.from_numpy(img).to(img_inp.dtype).to(img_inp.device)
        return img

    font = cv2.FONT_HERSHEY_PLAIN
    pos = (0, 0)
    font_thickness = 2
    text_color = (255, 255, 255)
    text_color_bg = (0, 0, 0)
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    img = cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    img = cv2.putText(
        img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness
    )
    if isinstance(img_inp, torch.Tensor):
        img = torch.from_numpy(img).to(img_inp.dtype).to(img_inp.device)
    return img


def load_rosetta_stats():
    stats_path = "/mnt/home/chpatel/runs/per_id_rosetta_deploy14_loop1_UD7_dummy/stats/"
    keys = ['avatar2hmc_Rt', 'avatar2hmc_rvec', 'avatar2hmc_tvec', 'avatar2hmc_rot6d', 'expression']
    stats = {}
    for key in keys:
        dct = torch.load(stats_path + key + '_stat.pt')
        # dct = {k: v.nan_to_num(1.0).float() for k, v in dct.items()}

        # Replace nan values in std with aproximate good values
        if key == 'avatar2hmc_rot6d':
            dct['std'][0] = 0.0057
        if key == 'avatar2hmc_Rt':
            dct['std'][0, 0] = 0.0057
        dct = {k: v.float() for k, v in dct.items()}
        stats[key] = dct

        # load covar matrix for expression
        if key == 'expression':
            Cinv = torch.load(
                "/mnt/home/leochli/CARE_UE/mixed_encodings/umvp_1e-6_encodings_inv_cov.pt"
            )
            W = torch.linalg.cholesky(Cinv)
            stats[key]['WT'] = W.T

    # perid stats
    perid_stats = torch.load(
        "/mnt/home/shaojieb/CARE_fix2/runs/mugsy_codes/encodings_16k_mugsy_arcata_perid_stats.pt"
    )

    def totorch(x):
        return {k: torch.from_numpy(v).float() for k, v in x.items()}

    stats['perid_stats'] = {}
    for k, v in perid_stats.items():
        assert k[:6] not in stats['perid_stats']
        stats['perid_stats'][k[:6]] = totorch(v)
    return stats


def gradient_postprocess(grad, p=7.0):
    ep = 2.718281828459**p
    g1 = torch.cat([grad.abs().log() / p, grad.sign()], -1)
    g2 = torch.cat([torch.ones_like(grad) * -1, grad * ep], -1)
    pick = grad.abs() > 1.0 / ep
    pick = torch.cat([pick, pick], -1)
    return torch.where(pick, g1, g2)


def get_angle_between_rot(a, b):
    assert a.shape[-1] == b.shape[-1]
    if a.shape[-1] == 3:
        a = rvec_to_R(a)
        b = rvec_to_R(b)
    elif a.shape[-1] == 6:
        a = rot6d_to_rotmat(a)
        b = rot6d_to_rotmat(b)
    else:
        raise NotImplementedError

    d = a @ b.transpose(-2, -1)
    cos_theta = (d.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) - 1) / 2
    cos_theta = cos_theta.clamp(-1, 1)
    return torch.arccos(cos_theta) * (180 / torch.pi)


def optimizer_with_careful_weight_decay(model, learning_rate, weight_decay, optim_type):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
    blacklist_weight_modules = (
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.PReLU,
    )
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = f'{mn}.{pn}' if mn else pn  # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # Store everys special parameter names here which don't belong to blacklist
    # or whitelist. These parameters are mostly nn.Parameters, nn.ParameterList
    # or nn.ParameterDict. In my case, these are mostly positional embeddings.
    special_cases = [
        'model.view_pos_emb',
        'model.view_pos_emb.0',
        'model.view_pos_emb.1',
        'model.view_pos_emb.2',
        'model.view_pos_emb.3',
        'model.view_pos_emb.4',
        'model.kp_pos_emb',
        'model.patch_pos_emb',
        'model.cls_token',
        'model.cond_pos_emb',
        'model.cond_pos_emb.0',
        'model.cond_pos_emb.1',
        'model.cond_pos_emb.2',
        'model.cond_pos_emb.3',
        'model.cond_pos_emb.4',
        'model.cond_pos_emb.4_0',
        'model.cond_pos_emb.4_1',
        'model.cond_pos_emb.3_0',
        'model.cond_pos_emb.3_1',
        'model.cond_pos_emb.2_0',
        'model.cond_pos_emb.2_1',
        'model.cond_pos_emb.1_0',
        'model.cond_pos_emb.1_1',
        'model.cond_pos_emb.0_0',
        'model.cond_pos_emb.0_1',
        'model.inp_pos_emb',
        'model.img_encoder.pos_embed',
        # 'model.param',
    ]
    temp = []
    for s in special_cases:
        temp.append(s.replace('model.', 'model.mod1.'))
        temp.append(s.replace('model.', 'model.mod2.'))
        temp.append(s.replace('model.', 'model.st_model.model.'))
        temp.append(s.replace('model.', 'model.sdm_model.model.'))
        temp.append(s.replace('model.', 'model.rgb_model.'))
        temp.append(s.replace('model.', 'slope_model.'))
        temp.append(s.replace('model.', 'expr_model.'))
        temp.append(s.replace('model.', 'model.uv_model.'))
    special_cases += temp

    temp = []
    for s in special_cases:
        temp.append(s.replace('model.', 'model.mod1.'))
        temp.append(s.replace('model.', 'model.mod2.'))
    special_cases += temp

    for k in special_cases:
        if k in param_dict:
            no_decay.add(k)
    for k in param_dict:
        if 'relative_position_bias_table' in k:
            no_decay.add(k)
        if k.endswith('.gamma'):
            no_decay.add(k)
        if 'view_pos_emb' in k:
            no_decay.add(k)
        if 'attn.pos_w' in k or 'attn.pos_h' in k:
            no_decay.add(k)
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert (
        len(inter_params) == 0
    ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    if optim_type == "adamw":
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, eps=1.0e-4)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, momentum=0.9, nesterov=True)
    elif optim_type == "radam":
        optim_class = load_class("care.strict.utils.radam.RAdamMultiTensor")
        optimizer = optim_class(optim_groups, lr=learning_rate, eps=1.0e-4)
    else:
        raise NotImplementedError

    return optimizer


def slice_batch(x, idx, ignore=[]):
    if isinstance(x, dict):
        ret = {}
        for k, v in x.items():
            if k in ignore:
                assert isinstance(v, list)
                ret[k] = [slice_batch(vi, idx, ignore) for vi in v]
            else:
                ret[k] = slice_batch(v, idx, ignore)
        return ret
    if isinstance(x, (int, float, str, bool)):
        return x
    if len(x) == 1:
        return x
    return x[idx]


def print_dict(d, prefix=None):
    if prefix is None:
        prefix = "r"

    if isinstance(d, dict):
        for k, v in d.items():
            print_dict(v, prefix + "/" + k)
    elif isinstance(d, (int, float, str, bool)):
        print(prefix, d)
    elif isinstance(d, list):
        print(prefix, "list", len(d))
        print_dict(d[0], prefix + "_0")
    elif isinstance(d, tuple):
        print(prefix, "tuple", len(d))
    elif hasattr(d, "shape"):
        print(prefix, d.shape)
    elif hasattr(d, "__str__"):
        print(prefix, d)
    else:
        print(prefix, type(d))


def get_num_iters(itr, max_iters, num_iters_schedule):
    if num_iters_schedule == "none":
        return max_iters
    elif num_iters_schedule == "step":
        return min(itr // 25000 + 1, max_iters)
    elif num_iters_schedule == "step2":
        return min(itr // 50000 + 2, max_iters)
    elif num_iters_schedule == "warmup":
        return 1 if itr < 2000 else max_iters
    elif num_iters_schedule == "random":
        return random.randrange(max_iters) + 1
    else:
        raise NotImplementedError


def resize2d(x, size, scale_factor=None):
    if size is None:
        assert scale_factor is not None
        size = tuple(int(s * scale_factor) for s in x.shape[-2:])
    size = tuple(size)
    assert len(size) == 2
    assert x.dim() >= 4

    if size[0] == x.shape[-2] and size[1] == x.shape[-1]:
        return x

    bdim = x.shape[:-3]
    x = x.flatten(0, -4)
    x = torch.nn.functional.interpolate(x, size, mode="bilinear", align_corners=False)
    x = x.unflatten(0, bdim)
    return x
