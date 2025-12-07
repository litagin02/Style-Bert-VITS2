"""
ベンチマークテスト用の共通ユーティリティ関数
"""

import random
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from style_bert_vits2.logging import logger


def save_benchmark_audio(
    audio_data: NDArray[np.float32],
    sample_rate: int,
    text: str,
    benchmark_name: str,
    suffix: str = "",
) -> None:
    """
    ベンチマーク用の音声ファイルを保存する共通関数。

    Args:
        audio_data: 音声データ (float32 配列)
        sample_rate: サンプリングレート
        text: 音声生成に使用したテキスト
        benchmark_name: ベンチマーク名（出力ディレクトリ名として使用）
        suffix: ファイル名のサフィックス（オプション）
    """
    try:
        from scipy.io import wavfile

        output_dir = Path(f"tests/wavs/{benchmark_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # ファイル名に使えない文字を置換
        safe_filename = (
            text.replace("/", "_")
            .replace("\\", "_")
            .replace(":", "")
            .replace("*", "")
            .replace("?", "")
            .replace("<", "")
            .replace(">", "")
            .replace("|", "")
            .replace('"', "")
            .replace("!", "")
        )

        # ファイル名が長すぎる場合は切り詰める
        max_filename_length = 70  # 拡張子とサフィックスを考慮した安全な長さ
        if len(safe_filename) > max_filename_length:
            safe_filename = safe_filename[: max_filename_length - 3] + "..."

        # サフィックスがある場合は追加
        if suffix:
            filename = f"{safe_filename}_{suffix}.wav"
        else:
            filename = f"{safe_filename}.wav"

        output_path = output_dir / filename

        # 16bit 整数に変換してWAVファイルとして保存
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(str(output_path), sample_rate, audio_int16)

        # ファイル保存成功をログ出力
        logger.debug(f"Audio saved: {output_path}")

    except ImportError:
        logger.warning("scipy is required to save audio files, skipping audio save")
    except Exception as ex:
        logger.error(f"Failed to save audio file: {ex}", exc_info=ex)


# デフォルトのランダムシード
DEFAULT_RANDOM_SEED = 42


def set_random_seeds(seed: int = DEFAULT_RANDOM_SEED) -> None:
    """
    すべてのランダム要素に固定シードを設定して再現性を確保する。

    Args:
        seed: ランダムシード値
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # CUDNN の決定的な動作を有効にする（速度は若干低下するが再現性が向上）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
