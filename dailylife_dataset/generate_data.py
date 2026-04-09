"""
Usage:
1) Generate all default keywords:
   python dailylife_dataset/generate_data.py

2) Generate only selected keywords:
   python dailylife_dataset/generate_data.py --keywords walk run throw

3) Regenerate even if same prompt-id npz exists:
   python dailylife_dataset/generate_data.py --keywords walk --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
from kimodo import load_model
from kimodo.exports.g1_npz_export import build_tiny_g1_npz_dict, prepare_npz_export_dict
from kimodo.tools import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DailyLife G1 dataset from prompts.txt files. "
            "Outputs npz files in the same folder as prompts.txt."
        )
    )
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=cfg.DEFAULT_KEYWORDS,
        help=f"Keywords to generate. Use 'all' for all configured keywords. Default: {cfg.DEFAULT_KEYWORDS}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if an existing <prompt_id>_*.npz already exists.",
    )
    return parser.parse_args()


def resolve_keywords(keywords: Iterable[str]) -> list[str]:
    all_keywords = list(cfg.KEYWORD_TO_PROMPTS.keys())
    normalized = [k.strip().lower() for k in keywords if k and k.strip()]
    if not normalized or normalized == ["all"]:
        return all_keywords

    invalid = [k for k in normalized if k not in cfg.KEYWORD_TO_PROMPTS]
    if invalid:
        raise ValueError(f"Unknown keywords: {invalid}. Valid options: {all_keywords}")
    return normalized


def load_prompts_with_line_id(path: Path) -> list[tuple[int, str]]:
    prompts: list[tuple[int, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_id, raw in enumerate(lines, start=1):
        prompt = raw.strip()
        if prompt:
            prompts.append((line_id, prompt))
    return prompts


def has_existing_prompt_id_npz(folder: Path, prompt_id: int) -> bool:
    return any(folder.glob(f"{prompt_id}_*.npz"))


def save_prompt_samples(
    output_for_npz: dict,
    prompt: str,
    prompt_id: int,
    out_dir: Path,
    num_samples: int,
) -> None:
    for sample_idx in range(num_samples):
        sample_id = sample_idx + 1
        filename = cfg.NPZ_NAME_PATTERN.format(prompt_id=prompt_id, sample_id=sample_id)
        out_path = out_dir / filename
        single = {
            k: (v[sample_idx] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == num_samples else v)
            for k, v in output_for_npz.items()
        }
        single["prompt"] = np.array(prompt)
        np.savez(out_path, **single)


def main() -> None:
    args = parse_args()
    keywords = resolve_keywords(args.keywords)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Keywords to generate: {keywords}")

    model, resolved_model = load_model(
        cfg.MODEL_NAME,
        device=device,
        default_family="Kimodo",
        return_resolved_name=True,
    )
    if "g1" not in resolved_model.lower():
        raise ValueError(f"This script requires a G1 model, got: {resolved_model}")

    print(
        f"Loaded model {resolved_model}; seed={cfg.SEED}, steps={cfg.DENOISING_STEPS}, "
        f"num_samples={cfg.NUM_SAMPLES_PER_PROMPT}, duration={cfg.DURATION_SECONDS}s"
    )

    seed_everything(cfg.SEED)
    num_frames = int(float(cfg.DURATION_SECONDS) * float(model.fps))

    for keyword in keywords:
        prompts_path = cfg.KEYWORD_TO_PROMPTS[keyword]
        if not prompts_path.exists():
            print(f"[SKIP] Missing prompts file for '{keyword}': {prompts_path}")
            continue

        out_dir = prompts_path.parent
        prompts = load_prompts_with_line_id(prompts_path)
        print(f"\n[{keyword}] prompts={len(prompts)} -> output dir: {out_dir}")

        for prompt_id, prompt in prompts:
            if (not args.force) and has_existing_prompt_id_npz(out_dir, prompt_id):
                print(f"  [SKIP] prompt line {prompt_id}: existing '{prompt_id}_*.npz' found")
                continue

            print(f"  [GEN ] line {prompt_id}: {prompt}")
            output = model(
                [prompt],
                [num_frames],
                constraint_lst=[],
                num_denoising_steps=cfg.DENOISING_STEPS,
                num_samples=cfg.NUM_SAMPLES_PER_PROMPT,
                multi_prompt=True,
                num_transition_frames=cfg.NUM_TRANSITION_FRAMES,
                post_processing=False,  # G1 export convention
                return_numpy=True,
            )
            output_for_npz = prepare_npz_export_dict(
                output,
                model.skeleton,
                fps=float(model.fps),
                device=str(model.device),
                root_quat_w_first=False,
                g1_as_mujoco_zup=True,
            )
            output_for_npz = build_tiny_g1_npz_dict(output_for_npz, model.skeleton)
            save_prompt_samples(
                output_for_npz=output_for_npz,
                prompt=prompt,
                prompt_id=prompt_id,
                out_dir=out_dir,
                num_samples=cfg.NUM_SAMPLES_PER_PROMPT,
            )
            print(f"  [DONE] line {prompt_id}: saved {cfg.NUM_SAMPLES_PER_PROMPT} samples")

    print("\nAll requested generations finished.")


if __name__ == "__main__":
    main()
