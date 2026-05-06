import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def npz_to_binary_image(npz_path) -> np.ndarray:
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    if "masks" not in data:
        raise KeyError(f"Missing 'masks' key in {npz_path}")

    masks = data["masks"]
    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks with shape (N, H, W), got {masks.shape} in {npz_path}"
        )

    if masks.shape[0] == 0:
        # Follow the producer format: empty masks may be saved as (0, 1, 1).
        binary = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        binary = np.any(masks.astype(bool), axis=0).astype(np.uint8) * 255

    return binary


def convert_directory(input_dir, output_dir) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    npz_files = sorted(input_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in npz_files:
        binary = npz_to_binary_image(npz_path)
        out_path = output_dir / f"{npz_path.stem}.png"
        Image.fromarray(binary, mode="L").save(out_path)
        print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert each hint_masks .npz file into one binary black-white image "
            "by unioning all masks in that file."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="/data/yjy_data/SAM2/hint_outputs_test/hint_masks",
        help="Directory containing hint mask .npz files (e.g. hint_masks/).",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/yjy_data/SAM2/hint_outputs_test/hint_mask_binary",
        help="Directory to save binary .png images.",
    )
    args = parser.parse_args()

    convert_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()