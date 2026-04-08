import argparse
from pathlib import Path

import numpy as np

from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton.registry import build_skeleton


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_with_dof{input_path.suffix}")


def _load_npz_dict(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _get_root_positions(data: dict, root_idx: int = 0) -> np.ndarray:
    if "root_positions" in data:
        return data["root_positions"]
    if "posed_joints" in data:
        posed = data["posed_joints"]
        if posed.ndim == 4:
            return posed[:, :, root_idx, :]
        if posed.ndim == 3:
            return posed[:, root_idx, :]
    raise KeyError("Could not infer root positions: missing both 'root_positions' and 'posed_joints'.")


def add_dof_keys(
    data: dict,
    *,
    mujoco_rest_zero: bool = False,
) -> dict:
    if "local_rot_mats" not in data:
        raise KeyError("Input NPZ must contain 'local_rot_mats'.")

    local_rot_mats = data["local_rot_mats"]
    root_positions = _get_root_positions(data, root_idx=0)

    # Compat: some older NPZs store a single [J, 3, 3] rotation frame.
    # In that case, repeat the frame across T from root_positions.
    if local_rot_mats.ndim == 3:
        if root_positions.ndim == 2:
            num_frames = root_positions.shape[0]
            local_rot_mats = np.repeat(local_rot_mats[None, ...], num_frames, axis=0)
        elif root_positions.ndim == 3:
            batch_size, num_frames = root_positions.shape[:2]
            local_rot_mats = np.broadcast_to(local_rot_mats, (batch_size, num_frames, *local_rot_mats.shape))
        else:
            raise ValueError(
                f"Unsupported root_positions ndim={root_positions.ndim} for local_rot_mats ndim=3."
            )

    if local_rot_mats.ndim not in (4, 5):
        raise ValueError(
            f"Unsupported local_rot_mats shape {local_rot_mats.shape}. "
            "Expected [T,J,3,3], [B,T,J,3,3], or legacy [J,3,3]."
        )
    if root_positions.ndim not in (2, 3):
        raise ValueError(
            f"Unsupported root_positions shape {root_positions.shape}. Expected [T,3] or [B,T,3]."
        )

    original_unbatched = local_rot_mats.ndim == 4
    if local_rot_mats.ndim == 4:
        local_rot_mats = local_rot_mats[None, ...]
    if root_positions.ndim == 2:
        root_positions = root_positions[None, ...]

    if local_rot_mats.shape[0] != root_positions.shape[0] or local_rot_mats.shape[1] != root_positions.shape[1]:
        raise ValueError(
            "local_rot_mats and root_positions have inconsistent batch/time dims: "
            f"{local_rot_mats.shape[:2]} vs {root_positions.shape[:2]}"
        )

    skeleton = build_skeleton(34)  # G1
    converter = MujocoQposConverter(skeleton)
    qpos = converter.dict_to_qpos(
        {"local_rot_mats": local_rot_mats, "root_positions": root_positions},
        numpy=True,
        mujoco_rest_zero=mujoco_rest_zero,
    )
    root_trans = qpos[..., :3]
    root_trans_offset = root_trans - root_trans[:, [0], :]
    root_rot = qpos[..., 3:7]
    dof = qpos[..., 7:]

    if original_unbatched:
        root_trans_offset = root_trans_offset[0]
        root_rot = root_rot[0]
        dof = dof[0]

    out = dict(data)
    out["root_trans_offset"] = root_trans_offset
    out["root_rot"] = root_rot
    out["dof"] = dof
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add G1 MuJoCo-style root_trans_offset / root_rot / dof keys to a Kimodo NPZ."
    )
    parser.add_argument("input_npz", type=Path, help="Path to input Kimodo NPZ.")
    parser.add_argument(
        "--output_npz",
        type=Path,
        default=None,
        help="Path to output NPZ. Defaults to <input_stem>_with_dof.npz",
    )
    parser.add_argument(
        "--mujoco-rest-zero",
        action="store_true",
        help="Export joint DoF relative to MuJoCo rest pose (q=0 at rest).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_npz = args.output_npz or _default_output_path(args.input_npz)

    data = _load_npz_dict(args.input_npz)
    out = add_dof_keys(data, mujoco_rest_zero=args.mujoco_rest_zero)
    np.savez(output_npz, **out)

    print(f"Saved: {output_npz}")
    print(f"Added keys: root_trans_offset {out['root_trans_offset'].shape}, root_rot {out['root_rot'].shape}, dof {out['dof'].shape}")


if __name__ == "__main__":
    main()
