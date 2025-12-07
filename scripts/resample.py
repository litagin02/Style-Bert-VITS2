import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import librosa
import pyloudnorm as pyln
import soundfile
from numpy.typing import NDArray
from tqdm import tqdm

from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


DEFAULT_BLOCK_SIZE: float = 0.400  # seconds

# Supported audio formats that will be converted to .wav
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".opus", ".m4a"}


class BlockSizeException(Exception):
    pass


def update_list_file_extensions(list_file: Path) -> int:
    """
    Update audio file extensions in a list file to .wav.

    List file format: path|speaker|language|text (or similar pipe-delimited format)
    The first field is the audio file path.

    Args:
        list_file: Path to the list file (e.g., esd.list)

    Returns:
        Number of lines updated
    """
    if not list_file.exists():
        return 0

    # Build regex pattern to match audio extensions at end of first field
    # Matches: path/to/audio.mp3|... or path\to\audio.ogg|...
    extensions_pattern = "|".join(
        re.escape(ext) for ext in SUPPORTED_AUDIO_EXTENSIONS if ext != ".wav"
    )
    # Pattern: (start or after separator)(path)(extension)(pipe)
    pattern = re.compile(rf"^([^|]*?)({extensions_pattern})(\|)", re.IGNORECASE)

    updated_count = 0
    lines = []

    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            new_line, num_subs = pattern.subn(r"\1.wav\3", line)
            if num_subs > 0:
                updated_count += 1
            lines.append(new_line)

    if updated_count > 0:
        with open(list_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        logger.info(f"Updated {updated_count} paths in {list_file}")

    return updated_count


def normalize_audio(data: NDArray[Any], sr: int):
    meter = pyln.Meter(sr, block_size=DEFAULT_BLOCK_SIZE)  # create BS.1770 meter
    try:
        loudness = meter.integrated_loudness(data)
    except ValueError as e:
        raise BlockSizeException(e)

    data = pyln.normalize.loudness(data, loudness, -23.0)
    return data


def resample(
    file: Path,
    input_dir: Path,
    output_dir: Path,
    target_sr: int,
    normalize: bool,
    trim: bool,
):
    """
    fileを読み込んで、target_srなwavファイルに変換して、
    output_dirの中に、input_dirからの相対パスを保つように保存する
    """
    try:
        # librosaが読めるファイルかチェック
        # wav以外にもmp3やoggやflacなども読める
        wav: NDArray[Any]
        sr: int
        wav, sr = librosa.load(file, sr=target_sr)
        if normalize:
            try:
                wav = normalize_audio(wav, sr)
            except BlockSizeException:
                print("")
                logger.info(
                    f"Skip normalize due to less than {DEFAULT_BLOCK_SIZE} second audio: {file}"
                )
        if trim:
            wav, _ = librosa.effects.trim(wav, top_db=30)
        relative_path = file.relative_to(input_dir)
        # ここで拡張子が.wav以外でも.wavに置き換えられる
        output_path = output_dir / relative_path.with_suffix(".wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(output_path, wav, sr)
    except Exception as e:
        logger.warning(f"Cannot load file, so skipping: {file}, {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sr",
        type=int,
        default=44100,
        help="sampling rate",
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="path to source dir",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="path to target dir",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=4,
        help="cpu_processes",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="loudness normalize audio",
    )
    parser.add_argument(
        "--trim",
        action="store_true",
        default=False,
        help="trim silence (start and end only)",
    )
    parser.add_argument(
        "--list_dir",
        "-l",
        type=str,
        default=None,
        help="Directory containing list files (e.g., esd.list) to update audio extensions. "
        "If not specified, uses parent directory of output_dir.",
    )
    args = parser.parse_args()

    if args.num_processes == 0:
        processes = cpu_count() - 2 if cpu_count() > 4 else 1
    else:
        processes: int = args.num_processes

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logger.info(f"Resampling {input_dir} to {output_dir}")
    sr = int(args.sr)
    normalize: bool = args.normalize
    trim: bool = args.trim

    # 後でlibrosaに読ませて有効な音声ファイルかチェックするので、全てのファイルを取得
    original_files = [f for f in input_dir.rglob("*") if f.is_file()]

    if len(original_files) == 0:
        logger.error(f"No files found in {input_dir}")
        raise ValueError(f"No files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=processes) as executor:
        futures = [
            executor.submit(resample, file, input_dir, output_dir, sr, normalize, trim)
            for file in original_files
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(original_files),
            file=SAFE_STDOUT,
            dynamic_ncols=True,
        ):
            pass

    logger.info("Resampling Done!")

    # Update list files to use .wav extensions
    list_dir = Path(args.list_dir) if args.list_dir else output_dir.parent
    if list_dir.exists():
        list_files = list(list_dir.glob("*.list"))
        if list_files:
            logger.info(
                f"Updating audio extensions in list files: {[f.name for f in list_files]}"
            )
            for list_file in list_files:
                update_list_file_extensions(list_file)
