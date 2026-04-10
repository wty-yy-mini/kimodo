"""Pack DailyLife filtered NPZ motions into one dataset file.

Output format:
    {
      "type_keyword_{name}": {
         "root_positions": np.ndarray[float32] (T, 3),
         "dof": np.ndarray[float32] (T, D),
         "root_rot": np.ndarray[float32] (T, 4),
         "fps": int,
         "prompt": str,
      },
      ...
    }
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PATH_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PATH_DIR / "dailylife_data_filter"
DEFAULT_OUTPUT_PKL = PATH_DIR / "dailylife_data_filter.pkl"


class NpzToJoblibPacker:
    """Pack categorized NPZ motion files into one dataset dictionary.

    Files are read from ``<type>/<keyword>/<name>.npz`` and packed with keys
    in ``type_keyword_{name}`` format.
    """

    def __init__(self, input_dir: Path, output_path: Path, strict_duplicate: bool = False) -> None:
        """Initialize packer configuration.

        Args:
            input_dir: Root directory containing categorized NPZ files.
            output_path: Destination dataset file path.
            strict_duplicate: Whether to fail on duplicate packed keys.
        """
        self.input_dir = input_dir
        self.output_path = output_path
        self.strict_duplicate = strict_duplicate

    @staticmethod
    def _to_python_scalar(x: Any) -> Any:
        """Convert NumPy scalar-like values to native Python scalars.

        Args:
            x: Candidate scalar or array value loaded from NPZ.

        Returns:
            A Python scalar when ``x`` is NumPy scalar-like, else the original value.
        """
        if isinstance(x, np.ndarray) and x.shape == ():
            return x.item()
        if isinstance(x, np.generic):
            return x.item()
        return x

    @classmethod
    def _normalize_prompt(cls, value: Any) -> str:
        """Normalize an NPZ prompt field to string.

        Args:
            value: Raw prompt value from NPZ.

        Returns:
            Decoded and stringified prompt text.
        """
        value = cls._to_python_scalar(value)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @classmethod
    def _normalize_fps(cls, value: Any) -> int:
        """Normalize an NPZ fps field to integer.

        Args:
            value: Raw fps value from NPZ.

        Returns:
            Rounded integer fps value.
        """
        value = cls._to_python_scalar(value)
        return int(round(float(value)))

    @staticmethod
    def _safe_token(name: str) -> str:
        """Sanitize a token for packed key composition.

        Args:
            name: Raw token text.

        Returns:
            Compact token with alphanumerics and underscores.
        """
        token = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
        token = re.sub(r"_+", "_", token).strip("_")
        return token

    @classmethod
    def _load_motion(cls, npz_path: Path) -> dict[str, Any]:
        """Load one NPZ motion file into normalized motion fields.

        Args:
            npz_path: NPZ motion file path.

        Returns:
            Dictionary with root_positions, dof, root_rot, fps and prompt.
        """
        with np.load(npz_path, allow_pickle=True) as data:
            root_positions = np.asarray(data["root_positions"], dtype=np.float32)
            dof = np.asarray(data["dof"], dtype=np.float32)
            root_rot = np.asarray(data["root_rot"], dtype=np.float32)
            fps = cls._normalize_fps(data["fps"])
            prompt = cls._normalize_prompt(data["prompt"])

        return {
            "root_positions": root_positions,
            "dof": dof,
            "root_rot": root_rot,
            "fps": fps,
            "prompt": prompt,
        }

    def pack(self) -> dict[str, dict[str, Any]]:
        """Pack all NPZ files under input directory into one dictionary.

        Returns:
            Mapping from ``type_keyword_{name}`` keys to motion dictionaries.
        """
        packed: dict[str, dict[str, Any]] = {}
        for npz_path in sorted(self.input_dir.glob("*/*/*.npz")):
            rel = npz_path.relative_to(self.input_dir)
            motion_type = self._safe_token(rel.parts[0])
            keyword = self._safe_token(rel.parts[1])
            name = self._safe_token(npz_path.stem)
            base_key = f"{motion_type}_{keyword}_{name}"

            key = base_key
            if key in packed and not self.strict_duplicate:
                suffix = 2
                while f"{base_key}_{suffix}" in packed:
                    suffix += 1
                key = f"{base_key}_{suffix}"

            packed[key] = self._load_motion(npz_path)

        return packed

    def save(self) -> None:
        """Pack motions and save them into a joblib dataset file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        packed = self.pack()
        joblib.dump(packed, self.output_path)

        total_frames = sum(int(v["root_positions"].shape[0]) for v in packed.values())
        fps_values = sorted({int(v["fps"]) for v in packed.values()})
        print(f"Packed motions: {len(packed)}")
        print(f"Total frames: {total_frames}")
        print(f"FPS values: {fps_values}")
        print(f"Output: {self.output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset packing.

    Returns:
        Parsed command-line options for input, output, and duplicate handling.
    """
    parser = argparse.ArgumentParser(
        description="Pack filtered DailyLife NPZ files into one visualization-friendly joblib dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Root folder containing <type>/<keyword>/*.npz (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PKL,
        help=f"Output dataset path (default: {DEFAULT_OUTPUT_PKL}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately on duplicate output keys (default behavior adds index suffix).",
    )
    return parser.parse_args()


def main() -> None:
    """Run CLI packing workflow and write the output dataset file."""
    args = parse_args()
    packer = NpzToJoblibPacker(
        input_dir=args.input_dir.resolve(),
        output_path=args.output.resolve(),
        strict_duplicate=args.strict,
    )
    packer.save()


if __name__ == "__main__":
    main()
