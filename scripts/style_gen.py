import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import librosa
import numpy as np
import torch
from config import get_config
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from pyannote.audio import Inference
from pyannote.audio.models.embedding.wespeaker import WeSpeakerResNet34
from tqdm import tqdm

from style_bert_vits2.logging import logger
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.utils.stdout_wrapper import SAFE_STDOUT


config = get_config()

# 今の pyannote.audio では Model.from_pretrained が torch.load でバグるため、
# 直接 load_from_checkpoint を呼ぶ
model_file = hf_hub_download(
    repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
    filename="pytorch_model.bin",
)
model = WeSpeakerResNet34.load_from_checkpoint(model_file, weights_only=False)
inference = Inference(model, window="whole")
device = torch.device(config.device)
inference.to(device)


class NaNValueError(ValueError):
    """カスタム例外クラス。NaN値が見つかった場合に使用されます。"""


# 推論時にインポートするために短いが関数を書く
def get_style_vector(wav_path: str) -> NDArray[Any]:
    # Windows では torchaudio (torchcodec) の読み込みに失敗することがあるため、
    # librosa で読み込んでから pyannote.audio に渡す
    # sr=None で元のサンプリングレートを維持
    wav, sr = librosa.load(wav_path, sr=None)
    wav_tensor = torch.from_numpy(wav).float()

    # librosa.load (mono=True default) returns (time,)
    # pyannote expects (channel, time)
    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    elif wav_tensor.ndim == 2:
        # librosa with mono=False returns (channel, time), so this path might not be reached with default defaults
        # but if it were (time, channel), we would transpose.
        # Librosa is usually (channel, time) if multi-channel.
        pass

    return inference({"waveform": wav_tensor, "sample_rate": sr})  # type: ignore


def save_style_vector(wav_path: str):
    try:
        style_vec = get_style_vector(wav_path)
    except Exception as e:
        print("\n")
        logger.error(f"Error occurred with file: {wav_path}, Details:\n{e}\n")
        raise
    # 値にNaNが含まれていると悪影響なのでチェックする
    if np.isnan(style_vec).any():
        print("\n")
        logger.warning(f"NaN value found in style vector: {wav_path}")
        raise NaNValueError(f"NaN value found in style vector: {wav_path}")
    np.save(f"{wav_path}.npy", style_vec)  # `test.wav` -> `test.wav.npy`


def process_line(line: str):
    wav_path = line.split("|")[0]
    try:
        save_style_vector(wav_path)
        return line, None
    except NaNValueError:
        return line, "nan_error"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--num_processes", type=int, default=config.num_processes)
    args, _ = parser.parse_known_args()
    config_path: str = args.config
    num_processes: int = args.num_processes

    hps = HyperParameters.load_from_json(config_path)

    training_lines: list[str] = []
    with open(hps.data.training_files, encoding="utf-8") as f:
        training_lines.extend(f.readlines())
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        training_results = list(
            tqdm(
                executor.map(process_line, training_lines),
                total=len(training_lines),
                file=SAFE_STDOUT,
                dynamic_ncols=True,
            )
        )
    ok_training_lines = [line for line, error in training_results if error is None]
    nan_training_lines = [
        line for line, error in training_results if error == "nan_error"
    ]
    if nan_training_lines:
        nan_files = [line.split("|")[0] for line in nan_training_lines]
        logger.warning(
            f"Found NaN value in {len(nan_training_lines)} files: {nan_files}, so they will be deleted from training data."
        )

    val_lines: list[str] = []
    with open(hps.data.validation_files, encoding="utf-8") as f:
        val_lines.extend(f.readlines())

    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        val_results = list(
            tqdm(
                executor.map(process_line, val_lines),
                total=len(val_lines),
                file=SAFE_STDOUT,
                dynamic_ncols=True,
            )
        )
    ok_val_lines = [line for line, error in val_results if error is None]
    nan_val_lines = [line for line, error in val_results if error == "nan_error"]
    if nan_val_lines:
        nan_files = [line.split("|")[0] for line in nan_val_lines]
        logger.warning(
            f"Found NaN value in {len(nan_val_lines)} files: {nan_files}, so they will be deleted from validation data."
        )

    with open(hps.data.training_files, "w", encoding="utf-8") as f:
        f.writelines(ok_training_lines)

    with open(hps.data.validation_files, "w", encoding="utf-8") as f:
        f.writelines(ok_val_lines)

    ok_num = len(ok_training_lines) + len(ok_val_lines)

    logger.info(f"Finished generating style vectors! total: {ok_num} npy files.")
