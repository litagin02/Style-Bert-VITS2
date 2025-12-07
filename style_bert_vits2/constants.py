from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from style_bert_vits2.utils.strenum import StrEnum


# Style-Bert-VITS2 のバージョン
try:
    VERSION = version("style-bert-vits2")
except PackageNotFoundError:
    VERSION = "0.0.0"

# Style-Bert-VITS2 のベースディレクトリ
BASE_DIR = Path(__file__).parent.parent


# 利用可能な言語
# v3.0.0 以降は日本語 (JP) のみサポート
class Languages(StrEnum):
    JP = "JP"


# 言語ごとのデフォルトの BERT モデルのパス
DEFAULT_BERT_MODEL_PATHS = {
    Languages.JP: "ku-nlp/deberta-v2-large-japanese-char-wwm",
}

# デフォルトの WavLM (SLM) モデルのパス (学習時のみ使用)
DEFAULT_SLM_MODEL = "microsoft/wavlm-base-plus"

# 言語ごとのデフォルトの BERT モデル (ONNX 版) のパス
DEFAULT_ONNX_BERT_MODEL_PATHS = {
    Languages.JP: BASE_DIR / "bert" / "deberta-v2-large-japanese-char-wwm-onnx",
}

# デフォルトのユーザー辞書ディレクトリ
## style_bert_vits2.nlp.japanese.user_dict モジュールのデフォルト値として利用される
## ライブラリとしての利用などで外部のユーザー辞書を指定したい場合は、user_dict 以下の各関数の実行時、引数に辞書データファイルのパスを指定する
DEFAULT_USER_DICT_DIR = BASE_DIR / "dict_data"

# デフォルトの推論パラメータ
DEFAULT_STYLE = "Neutral"
DEFAULT_STYLE_WEIGHT = 1.0
DEFAULT_SDP_RATIO = 0.2
DEFAULT_NOISE = 0.6
DEFAULT_NOISEW = 0.8
DEFAULT_LENGTH = 1.0
DEFAULT_LINE_SPLIT = True
DEFAULT_SPLIT_INTERVAL = 0.5
DEFAULT_ASSIST_TEXT_WEIGHT = 0.7
DEFAULT_ASSIST_TEXT_WEIGHT = 1.0

# Gradio のテーマ
## Built-in theme: "default", "base", "monochrome", "soft", "glass"
## See https://huggingface.co/spaces/gradio/theme-gallery for more themes
GRADIO_THEME = "NoCrypt/miku"
