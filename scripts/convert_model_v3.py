#!/usr/bin/env python
"""
Convert pre-3.x JP-Extra models to v3.0 format.

This script converts the embedding layers from the multilingual format (v2) to
the Japanese-only format (v3):

- enc_p.emb.weight: [~108, hidden] -> [53, hidden] (JP symbols only)
- enc_p.tone_emb.weight: [12, hidden] -> [2, hidden] (JP tones only)
- enc_p.language_emb.weight: [3, hidden] -> [1, hidden] (JP language only)

Usage:
    # Single file
    python scripts/convert_model_v3.py -i model.safetensors -o model_v3.safetensors

    # Directory (flat)
    python scripts/convert_model_v3.py -i pretrained_jp_extra/ -o pretrained_v3/

    # Convert all models in model_assets (adds _v3 suffix to each file)
    python scripts/convert_model_v3.py --model_assets model_assets/
    # e.g., model_assets/jvnv-F1-jp/model.safetensors -> model_assets/jvnv-F1-jp/model_v3.safetensors

    # Convert all models in model_assets to a new directory
    python scripts/convert_model_v3.py --model_assets model_assets/ -o model_assets_v3/
"""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from style_bert_vits2.nlp.symbols import (
    LANGUAGE_ID_MAP_V2,
    LANGUAGE_TONE_START_MAP_V2,
    NUM_TONES_V2,
    NUM_TONES_V3,
    SYMBOLS_V2,
    SYMBOLS_V3,
)


def build_symbol_index_mapping() -> dict[int, int]:
    """
    Build mapping from v2 symbol indices to v3 symbol indices.
    Only symbols that exist in both v2 and v3 are mapped.

    Returns:
        dict mapping v2 index -> v3 index
    """
    v2_to_v3 = {}

    for v3_idx, symbol in enumerate(SYMBOLS_V3):
        if symbol in SYMBOLS_V2:
            v2_idx = SYMBOLS_V2.index(symbol)
            v2_to_v3[v2_idx] = v3_idx

    return v2_to_v3


def convert_model(input_path: Path, output_path: Path, verbose: bool = True) -> None:
    """
    Convert a single model file from v2 to v3 format.

    Args:
        input_path: Path to input safetensors file
        output_path: Path to output safetensors file
        verbose: Whether to print progress information
    """
    if verbose:
        print(f"Converting: {input_path} -> {output_path}")

    # Load all tensors from the input file
    with safe_open(input_path, framework="pt", device="cpu") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    # Check if conversion is needed
    emb_key = "enc_p.emb.weight"
    if emb_key not in tensors:
        if verbose:
            print(f"  Skipping: No {emb_key} found (not a generator model)")
        # Just copy the file as-is
        save_file(tensors, output_path)
        return

    old_emb = tensors[emb_key]
    vocab_size = old_emb.shape[0]

    if vocab_size == len(SYMBOLS_V3):
        if verbose:
            print(f"  Skipping: Already v3 format (vocab_size={vocab_size})")
        save_file(tensors, output_path)
        return

    if vocab_size != len(SYMBOLS_V2):
        if verbose:
            print(
                f"  Warning: Unknown vocab size {vocab_size}, expected {len(SYMBOLS_V2)} (v2) or {len(SYMBOLS_V3)} (v3)"
            )
            print(f"  Copying file without conversion...")
        save_file(tensors, output_path)
        return

    hidden_channels = old_emb.shape[1]
    if verbose:
        print(f"  Vocab: {vocab_size} -> {len(SYMBOLS_V3)}")
        print(f"  Hidden channels: {hidden_channels}")

    # Build symbol mapping
    v2_to_v3 = build_symbol_index_mapping()

    # Convert enc_p.emb.weight
    new_emb = torch.zeros(len(SYMBOLS_V3), hidden_channels, dtype=old_emb.dtype)
    for v2_idx, v3_idx in v2_to_v3.items():
        new_emb[v3_idx] = old_emb[v2_idx]
    tensors[emb_key] = new_emb
    if verbose:
        print(
            f"  Converted {emb_key}: [{vocab_size}, {hidden_channels}] -> [{len(SYMBOLS_V3)}, {hidden_channels}]"
        )

    # Convert enc_p.tone_emb.weight if present
    tone_emb_key = "enc_p.tone_emb.weight"
    if tone_emb_key in tensors:
        old_tone_emb = tensors[tone_emb_key]
        if old_tone_emb.shape[0] == NUM_TONES_V2:
            # JP tones are at indices 6-7 in v2 (after ZH tones 0-5)
            jp_tone_start = LANGUAGE_TONE_START_MAP_V2["JP"]  # 6
            new_tone_emb = old_tone_emb[
                jp_tone_start : jp_tone_start + NUM_TONES_V3
            ]  # indices 6-7
            tensors[tone_emb_key] = new_tone_emb
            if verbose:
                print(
                    f"  Converted {tone_emb_key}: [{NUM_TONES_V2}, {old_tone_emb.shape[1]}] -> [{NUM_TONES_V3}, {new_tone_emb.shape[1]}]"
                )
        else:
            if verbose:
                print(
                    f"  Skipping {tone_emb_key}: unexpected shape {old_tone_emb.shape}"
                )

    # Convert enc_p.language_emb.weight if present
    lang_emb_key = "enc_p.language_emb.weight"
    if lang_emb_key in tensors:
        old_lang_emb = tensors[lang_emb_key]
        num_languages_v2 = len(LANGUAGE_ID_MAP_V2)  # 3
        if old_lang_emb.shape[0] == num_languages_v2:
            # JP is at index 1 in v2
            jp_lang_idx = LANGUAGE_ID_MAP_V2["JP"]  # 1
            new_lang_emb = old_lang_emb[jp_lang_idx : jp_lang_idx + 1]  # index 1 only
            tensors[lang_emb_key] = new_lang_emb
            if verbose:
                print(
                    f"  Converted {lang_emb_key}: [{num_languages_v2}, {old_lang_emb.shape[1]}] -> [1, {new_lang_emb.shape[1]}]"
                )
        else:
            if verbose:
                print(
                    f"  Skipping {lang_emb_key}: unexpected shape {old_lang_emb.shape}"
                )

    # Save converted model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output_path)

    if verbose:
        old_size = input_path.stat().st_size / 1024 / 1024
        new_size = output_path.stat().st_size / 1024 / 1024
        print(f"  Size: {old_size:.2f} MB -> {new_size:.2f} MB")


def convert_model_assets(
    assets_dir: Path, output_dir: Path | None, verbose: bool = True
) -> None:
    """
    Convert all models in a model_assets directory structure.

    Expected structure:
        model_assets/
        ├── model1/
        │   ├── config.json
        │   ├── model1.safetensors
        │   └── style_vectors.npy
        ├── model2/
        ...

    Args:
        assets_dir: Path to model_assets directory
        output_dir: Path to output directory (None = add _v3 suffix in same location)
        verbose: Whether to print progress information
    """
    # Find all safetensors files recursively
    safetensors_files = list(assets_dir.rglob("*.safetensors"))

    if not safetensors_files:
        print(f"No safetensors files found in {assets_dir}")
        return

    print(f"Found {len(safetensors_files)} safetensors files in {assets_dir}")

    for sf in safetensors_files:
        if output_dir is None:
            # Add _v3 suffix: model.safetensors -> model_v3.safetensors
            out_file = sf.with_stem(sf.stem + "_v3")
            convert_model(sf, out_file, verbose=verbose)
        else:
            # Preserve directory structure in output
            relative_path = sf.relative_to(assets_dir)
            out_file = output_dir / relative_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            convert_model(sf, out_file, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(
        description="Convert pre-3.x JP-Extra models to v3.0 format"
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input safetensors file or directory containing safetensors files",
    )
    parser.add_argument("--output", "-o", help="Output safetensors file or directory")
    parser.add_argument(
        "--model_assets",
        help="Convert all models in a model_assets directory (recursive). "
        "If --output is not specified, adds _v3 suffix to each file.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Print progress information (default: True)",
    )

    args = parser.parse_args()

    # Handle --model_assets mode
    if args.model_assets:
        assets_dir = Path(args.model_assets)
        if not assets_dir.is_dir():
            print(f"model_assets directory does not exist: {assets_dir}")
            return
        output_dir = Path(args.output) if args.output else None
        convert_model_assets(assets_dir, output_dir, verbose=args.verbose)
        print("\nConversion complete!")
        return

    # Handle --input mode
    if not args.input:
        parser.error("Either --input or --model_assets is required")

    if not args.output:
        parser.error("--output is required when using --input")

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        # Single file conversion
        convert_model(input_path, output_path, verbose=args.verbose)
    elif input_path.is_dir():
        # Directory conversion
        safetensors_files = list(input_path.glob("*.safetensors"))
        if not safetensors_files:
            print(f"No safetensors files found in {input_path}")
            return

        output_path.mkdir(parents=True, exist_ok=True)

        for sf in safetensors_files:
            out_file = output_path / sf.name
            convert_model(sf, out_file, verbose=args.verbose)
    else:
        print(f"Input path does not exist: {input_path}")
        return

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
