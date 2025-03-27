import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import soundfile as sf
import numpy as np
from typing import Union, Optional
from pathlib import Path

def separate_vocals_with_demucs(audio_path: Union[str, Path], output_path: Union[str, Path]) -> Optional[str]:
    # モデルのロード
    model = get_model('htdemucs')
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 音声を読み込む
    audio, sr = sf.read(audio_path)
    
    # ステレオに変換（モノラルの場合）
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio])
    else:
        audio = audio.T  # [channels, samples] の形式に変換
    
    # バッチ次元を追加し、明示的にfloat32型に変換
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    
    if torch.cuda.is_available():
        audio = audio.cuda()
    
    # 分離処理
    with torch.no_grad():
        sources = apply_model(model, audio, device=next(model.parameters()).device)
    
    # CPU に戻す
    sources = sources.cpu().numpy()
    
    # 分離した音声を取得 (vocals)
    vocals = sources[0, model.sources.index('vocals')]
    
    # 保存
    sf.write(output_path, vocals.T, sr)
    
    return str(output_path) if output_path else None

if __name__ == "__main__":
    input_file = "Data/row_data/toy_3_128K_aac merged.wav"
    output_file = "Data/row_data/buzz_lightyear_vocals_3.wav"
    separate_vocals_with_demucs(input_file, output_file)