"""DailyLife dataset generation entrypoint.

Usage:
1) Generate all default keywords:
   python dailylife_dataset/generate_data.py

2) Generate only selected keywords:
   python dailylife_dataset/generate_data.py --keywords walk run throw

3) Generate only one prompt line (id start from 1):
   python dailylife_dataset/generate_data.py --keywords jump --prompt-id 3

4) Regenerate even if same prompt-id npz exists:
   python dailylife_dataset/generate_data.py --keywords walk --overwrite
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
    """Parse command-line arguments for dataset generation.

    Returns:
        Parsed command-line namespace.
    """
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
        "--prompt-id",
        type=int,
        default=None,
        help="Generate only the specified non-empty prompt line id (Start from 1). Default: generate all prompts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even if an existing <prompt_id>_*.npz already exists.",
    )
    args = parser.parse_args()
    if args.prompt_id is not None and args.prompt_id <= 0:
        parser.error("--prompt-id must be a positive line id.")
    return args


def resolve_keywords(keywords: Iterable[str]) -> list[str]:
    """Resolve requested keywords against configured prompt groups.

    Args:
        keywords: Raw keyword values from command-line arguments.

    Returns:
        Normalized keyword list to generate.
    """
    all_keywords = list(cfg.KEYWORD_TO_PROMPTS.keys())
    normalized = [k.strip().lower() for k in keywords if k and k.strip()]
    if not normalized or normalized == ["all"]:
        return all_keywords

    invalid = [k for k in normalized if k not in cfg.KEYWORD_TO_PROMPTS]
    if invalid:
        raise ValueError(f"Unknown keywords: {invalid}. Valid options: {all_keywords}")
    return normalized


def load_prompts_with_line_id(path: Path) -> list[tuple[int, str]]:
    """Load non-empty prompts while preserving their source line ids.

    Args:
        path: Path to the prompts.txt file.

    Returns:
        List of ``(line_id, prompt)`` pairs for non-empty lines.
    """
    prompts: list[tuple[int, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_id, raw in enumerate(lines, start=1):
        prompt = raw.strip()
        if prompt:
            prompts.append((line_id, prompt))
    return prompts


def has_existing_prompt_id_npz(folder: Path, prompt_id: int) -> bool:
    """Check whether output files already exist for a prompt line id.

    Args:
        folder: Output directory to inspect.
        prompt_id: Prompt line id encoded in output filenames.

    Returns:
        ``True`` when matching ``<prompt_id>_*.npz`` files already exist.
    """
    return any(folder.glob(f"{prompt_id}_*.npz"))


def filter_prompts_by_id(prompts: list[tuple[int, str]], prompt_id: int | None) -> list[tuple[int, str]]:
    """Filter prompts to a single requested line id when provided.

    Args:
        prompts: Loaded ``(line_id, prompt)`` pairs.
        prompt_id: Requested non-empty prompt line id, or ``None`` for all prompts.

    Returns:
        Filtered prompt list matching the requested line id.
    """
    if prompt_id is None:
        return prompts
    return [(line_id, prompt) for line_id, prompt in prompts if line_id == prompt_id]


def save_prompt_samples(
    output_for_npz: dict,
    prompt: str,
    prompt_id: int,
    out_dir: Path,
    num_samples: int,
) -> None:
    """Save one generated prompt result into per-sample NPZ files.

    Args:
        output_for_npz: Export-ready batched arrays keyed by NPZ field name.
        prompt: Prompt text used for generation.
        prompt_id: Prompt line id encoded into output filenames.
        out_dir: Directory where NPZ files will be written.
        num_samples: Number of samples to save from the batched output.
    """
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
    """Run DailyLife prompt generation for the requested keyword groups.

    The generation loop can optionally restrict work to one non-empty prompt
    line id across the selected keywords.
    """
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
        f"num_samples={cfg.NUM_SAMPLES_PER_PROMPT}, default_duration={cfg.DURATION_SECONDS_DEFAULT}s"
    )

    seed_everything(cfg.SEED)

    for keyword in keywords:
        duration_seconds = cfg.get_duration_seconds(keyword)
        num_frames = int(float(duration_seconds) * float(model.fps))
        prompts_path = cfg.KEYWORD_TO_PROMPTS[keyword]
        if not prompts_path.exists():
            print(f"[SKIP] Missing prompts file for '{keyword}': {prompts_path}")
            continue

        out_dir = prompts_path.parent
        prompts = load_prompts_with_line_id(prompts_path)
        prompts = filter_prompts_by_id(prompts, args.prompt_id)
        print(
            f"\n[{keyword}] prompts={len(prompts)}, duration={duration_seconds}s, "
            f"num_frames={num_frames} -> output dir: {out_dir}"
        )

        if args.prompt_id is not None and not prompts:
            print(f"  [SKIP] prompt line {args.prompt_id}: not found in {prompts_path}")
            continue

        for prompt_id, prompt in prompts:
            if (not args.overwrite) and has_existing_prompt_id_npz(out_dir, prompt_id):
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
