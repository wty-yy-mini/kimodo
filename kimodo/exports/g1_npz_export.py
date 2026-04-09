"""Shared helpers for NPZ export payload preparation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .mujoco import MujocoQposConverter


def _is_g1_skeleton(skeleton: Any) -> bool:
    return "g1" in getattr(skeleton, "name", "").lower()


def _infer_root_positions(output: dict, skeleton: Any) -> Optional[np.ndarray]:
    root_positions = output.get("root_positions")
    if root_positions is not None:
        return root_positions

    posed_joints = output.get("posed_joints")
    if posed_joints is None:
        return None
    root_idx = skeleton.root_idx
    if posed_joints.ndim == 4:
        return posed_joints[:, :, root_idx, :]
    if posed_joints.ndim == 3:
        return posed_joints[:, root_idx, :]
    return None


def add_g1_dof_export_keys(
    output: dict,
    skeleton: Any,
    *,
    device: Optional[str] = None,
    root_quat_w_first: bool = True,
) -> dict:
    """Augment G1 outputs with root_trans_offset / dof / root_rot."""
    if not _is_g1_skeleton(skeleton):
        return output
    if "local_rot_mats" not in output:
        return output

    root_positions = _infer_root_positions(output, skeleton)
    if root_positions is None:
        return output

    converter = MujocoQposConverter(skeleton)
    local_rot_mats = output["local_rot_mats"]
    root_positions_in = root_positions
    added_batch = False
    if (
        isinstance(local_rot_mats, np.ndarray)
        and isinstance(root_positions_in, np.ndarray)
        and local_rot_mats.ndim == 4
        and root_positions_in.ndim == 2
    ):
        local_rot_mats = local_rot_mats[None, ...]
        root_positions_in = root_positions_in[None, ...]
        added_batch = True

    qpos = converter.dict_to_qpos(
        {"local_rot_mats": local_rot_mats, "root_positions": root_positions_in},
        device=device,
        numpy=True,
        root_quat_w_first=root_quat_w_first,
    )
    if added_batch and qpos.ndim == 3 and qpos.shape[0] == 1:
        qpos = qpos[0]

    out = dict(output)

    out["root_rot"] = qpos[..., 3:7]
    out["dof"] = qpos[..., 7:]
    return out


def convert_g1_npz_to_mujoco_zup(output: dict, skeleton: Any) -> dict:
    """Convert G1 motion tensors in an export dict from Kimodo y-up to MuJoCo z-up."""
    if not _is_g1_skeleton(skeleton):
        return output

    converter = MujocoQposConverter(skeleton)
    c = converter.kimodo_to_mujoco_matrix.detach().cpu().numpy()
    ct = c.T

    def _transform_vec(arr: np.ndarray) -> np.ndarray:
        return np.einsum("ij,...j->...i", c, arr)

    def _transform_rot(arr: np.ndarray) -> np.ndarray:
        return np.einsum("ij,...jk,kl->...il", c, arr, ct)

    out = dict(output)
    for key in ("posed_joints", "root_positions"):
        val = out.get(key)
        if isinstance(val, np.ndarray) and val.shape[-1] == 3:
            out[key] = _transform_vec(val)

    for key in ("global_rot_mats", "local_rot_mats"):
        val = out.get(key)
        if isinstance(val, np.ndarray) and val.shape[-2:] == (3, 3):
            out[key] = _transform_rot(val)

    out["coord_system"] = np.array("mujoco_zup_xforward_righthanded")
    return out


def prepare_npz_export_dict(
    output: dict,
    skeleton: Any,
    *,
    fps: Optional[float] = None,
    device: Optional[str] = None,
    root_quat_w_first: bool = True,
    g1_as_mujoco_zup: bool = True,
) -> dict:
    """Prepare a unified NPZ payload used by both CLI and UI exports."""
    out = add_g1_dof_export_keys(
        output,
        skeleton,
        device=device,
        root_quat_w_first=root_quat_w_first,
    )
    if g1_as_mujoco_zup:
        out = convert_g1_npz_to_mujoco_zup(out, skeleton)
    if fps is not None:
        out = dict(out)
        out["fps"] = np.array(float(fps), dtype=np.float32)
    return out


def build_tiny_g1_npz_dict(output: dict, skeleton: Any) -> dict:
    """Build tiny G1 NPZ payload with only root_positions/root_rot/dof/fps."""
    if not _is_g1_skeleton(skeleton):
        raise ValueError("--tiny-npz is only supported for G1 models.")

    required_keys = ("root_positions", "root_rot", "dof", "fps")
    missing = [k for k in required_keys if k not in output]
    if missing:
        raise KeyError(f"Tiny G1 NPZ requires keys {required_keys}, missing: {missing}")

    return {k: output[k] for k in required_keys}
