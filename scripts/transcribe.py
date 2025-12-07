# 何もしないダミー torchcodec モジュールを登録
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import torch
from config import get_path_config
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import WhisperProcessor, pipeline
from transformers.utils import import_utils

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


# torchcodec は、Windowsではimportした瞬間にエラーが出る呪いがあり、transformers の pipeline 実行時にこの呪いが発動する。
# しかし pyannote.audio が依存しているため、完全に無視するわけにもいかない。
# そこで transformers 内部で `is_torchcodec_available()` が呼ばれたときに False を返すように、定数を上書きする。

import_utils._torchcodec_available = False


def check_ffmpeg():
    """
    Check if ffmpeg is installed and available in PATH.
    If not found in system PATH, check ../lib/ffmpeg/bin (installer structure) and add to PATH if found.
    """
    if shutil.which("ffmpeg") is not None:
        return True

    # Check up one level (installer structure: Root/Install.bat, Root/lib/ffmpeg, Root/Style-Bert-VITS2/transcribe.py)
    local_ffmpeg_up = (
        Path(__file__).parent.parent / "lib" / "ffmpeg" / "bin" / "ffmpeg.exe"
    )
    if local_ffmpeg_up.exists():
        logger.info(f"Found local ffmpeg at {local_ffmpeg_up}, adding to PATH")
        os.environ["PATH"] = (
            str(local_ffmpeg_up.parent) + os.pathsep + os.environ["PATH"]
        )
        return True

    return False


# HF pipelineで進捗表示をするために必要なDatasetクラス
class LibrosaDataset(Dataset[np.ndarray]):
    def __init__(self, original_list: list[Path]) -> None:
        self.original_list = original_list

    def __len__(self) -> int:
        return len(self.original_list)

    def __getitem__(self, i: int) -> np.ndarray:
        file_path = self.original_list[i]
        audio, _ = librosa.load(file_path, sr=16_000)
        return audio


# HFのWhisperはファイルリストを与えるとバッチ処理ができて速い
def transcribe_files_with_hf_whisper(
    audio_files: list[Path],
    model_id: str,
    output_file: Path,
    model_name: str,
    language_id: int,
    input_dir: Path,
    initial_prompt: Optional[str] = None,
    language: str = "ja",
    batch_size: int = 16,
    num_beams: int = 1,
    no_repeat_ngram_size: int = 10,
    device: str = "cuda",
) -> None:
    processor: WhisperProcessor = WhisperProcessor.from_pretrained(model_id)
    generate_kwargs: dict[str, Any] = {
        "language": language,
        "do_sample": False,
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    logger.info(f"generate_kwargs: {generate_kwargs}, loading pipeline...")
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=batch_size,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device=device,
        trust_remote_code=True,
    )
    logger.info("Loaded pipeline")
    if initial_prompt is not None:
        # tokenizer might not have get_prompt_ids in some versions/models, fallback to basic encoding if needed
        # but standard WhisperTokenizer has it.
        prompt_ids: torch.Tensor = pipe.tokenizer.get_prompt_ids(
            initial_prompt, return_tensors="pt"
        ).to(device)
        generate_kwargs["prompt_ids"] = prompt_ids

    dataset = LibrosaDataset([f for f in audio_files])

    # Initialize tqdm here, just before processing starts
    pbar = tqdm(total=len(audio_files), file=SAFE_STDOUT, dynamic_ncols=True)

    for whisper_result, file in zip(
        pipe(dataset, generate_kwargs=generate_kwargs), audio_files
    ):
        text: str = whisper_result["text"]
        # なぜかテキストの最初に" {initial_prompt}"が入るので、文字の最初からこれを削除する
        if initial_prompt and text.startswith(f" {initial_prompt}"):
            text = text[len(f" {initial_prompt}") :]

        # Append to file immediately
        with open(output_file, "a", encoding="utf-8") as f:
            wav_rel_path = file.relative_to(input_dir)
            f.write(f"{wav_rel_path}|{model_name}|{language_id}|{text}\n")

        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--initial_prompt",
        type=str,
        default="こんにちは。元気、ですかー？ふふっ、私は……ちゃんと元気だよ！",
    )
    parser.add_argument(
        "--language", type=str, default="ja", choices=["ja", "en", "zh"]
    )
    parser.add_argument("--model_id", type=str, default="openai/whisper-large-v3-turbo")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=10)
    args = parser.parse_args()

    path_config = get_path_config()
    dataset_root = path_config.dataset_root

    model_name = str(args.model_name)

    input_dir = dataset_root / model_name / "raw"
    output_file = dataset_root / model_name / "esd.list"
    initial_prompt: str = args.initial_prompt
    initial_prompt = initial_prompt.strip('"')
    language: str = args.language
    device: str = args.device
    model_id: str = args.model_id
    batch_size: int = args.batch_size
    num_beams: int = args.num_beams
    no_repeat_ngram_size: int = args.no_repeat_ngram_size

    if not check_ffmpeg():
        logger.error(
            "FFmpeg is not found! Please install FFmpeg and add it to PATH, or place it in lib/ffmpeg/bin."
        )
        sys.exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Support multiple audio formats
    SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".opus", ".m4a"}
    audio_files = [
        f
        for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    audio_files = sorted(audio_files, key=lambda x: str(x))
    logger.info(f"Found {len(audio_files)} audio files")
    if len(audio_files) == 0:
        logger.warning(f"No audio files found in {input_dir}")
        sys.exit(1)

    if output_file.exists():
        logger.warning(f"{output_file} exists, backing up to {output_file}.bak")
        backup_path = output_file.with_name(output_file.name + ".bak")
        if backup_path.exists():
            logger.warning(f"{output_file}.bak exists, deleting...")
            backup_path.unlink()
        output_file.rename(backup_path)

    if language == "ja":
        language_id = Languages.JP.value
    elif language == "en":
        language_id = Languages.EN.value
    elif language == "zh":
        language_id = Languages.ZH.value
    else:
        raise ValueError(f"{language} is not supported.")

    logger.info(f"Loading HF Whisper model ({model_id})")

    transcribe_files_with_hf_whisper(
        audio_files=audio_files,
        model_id=model_id,
        output_file=output_file,
        model_name=model_name,
        language_id=language_id,
        input_dir=input_dir,
        initial_prompt=initial_prompt,
        language=language,
        batch_size=batch_size,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        device=device,
    )

    sys.exit(0)
