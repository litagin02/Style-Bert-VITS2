"""
anime-whisperを使用した音声文字起こしラッパースクリプト
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="anime-whisperを使用した音声文字起こし")
    parser.add_argument("--model_name", type=str, required=True, help="モデル名")
    parser.add_argument("--language", type=str, default="ja", help="言語 (ja/en/zh)")
    parser.add_argument("--device", type=str, default="cuda", help="デバイス (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=8, help="バッチサイズ")
    args = parser.parse_args()

    # transcribe.pyを呼び出す
    # anime-whisperはinitial_promptと相性が悪いため、空文字列を渡す
    cmd = [
        sys.executable,
        "transcribe.py",
        "--model_name", args.model_name,
        "--use_anime_whisper",
        "--initial_prompt", '""',  # 空文字列
        "--language", args.language,
        "--device", args.device,
        "--batch_size", str(args.batch_size),
    ]

    print("anime-whisperを使用して文字起こしを開始します...")
    print(f"コマンド: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())