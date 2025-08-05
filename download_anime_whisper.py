"""
anime-whisperモデルのダウンロードスクリプト
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from style_bert_vits2.logging import logger


def download_anime_whisper(cache_dir: Path = None):
    """anime-whisperモデルをダウンロード"""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    model_id = "litagin/anime-whisper"
    
    logger.info(f"Downloading anime-whisper model from {model_id}...")
    logger.info(f"Cache directory: {cache_dir}")
    
    try:
        # モデルをダウンロード
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
            max_workers=8,
        )
        
        logger.info(f"Model downloaded successfully to: {model_path}")
        
        # ダウンロードされたファイルを確認
        model_path = Path(model_path)
        if model_path.exists():
            files = list(model_path.rglob("*"))
            logger.info(f"Downloaded {len(files)} files")
            
            # 主要なファイルの存在を確認
            important_files = [
                "pytorch_model.bin",
                "config.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
            ]
            
            for file_name in important_files:
                if any(f.name == file_name for f in files):
                    logger.info(f"✓ {file_name} found")
                else:
                    logger.warning(f"✗ {file_name} not found")
        
        return model_path
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="anime-whisperモデルをダウンロード")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="キャッシュディレクトリ (デフォルト: ~/.cache/huggingface/hub)",
    )
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    
    model_path = download_anime_whisper(cache_dir)
    
    if model_path:
        logger.info("anime-whisperモデルのダウンロードが完了しました！")
        logger.info("transcribe_anime.pyを使用して文字起こしを実行できます。")
        return 0
    else:
        logger.error("モデルのダウンロードに失敗しました。")
        return 1


if __name__ == "__main__":
    sys.exit(main())