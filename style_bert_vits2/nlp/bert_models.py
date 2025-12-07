"""
Style-Bert-VITS2 の学習・推論に必要な BERT モデルをロード/取得するためのモジュール。

v3.0.0 以降は日本語 (JP) のみサポート。
ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき。
一度 load_model/tokenizer() で BERT モデルがロードされていれば、ライブラリ内部のどこからでもロード済みのモデル/トークナイザーを取得できる。
"""

from __future__ import annotations

import gc
import time
from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from style_bert_vits2.constants import DEFAULT_BERT_MODEL_PATHS, Languages
from style_bert_vits2.logging import logger


if TYPE_CHECKING:
    pass


# 各言語ごとのロード済みの BERT モデルを格納する辞書
__loaded_models: dict[Languages, PreTrainedModel] = {}

# 各言語ごとのロード済みの BERT トークナイザーを格納する辞書
__loaded_tokenizers: dict[
    Languages,
    Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
] = {}

# 各言語ごとの BERT モデルの現在の dtype を格納する辞書
__model_dtypes: dict[Languages, Optional["torch.dtype"]] = {}


def load_model(
    language: Languages,
    pretrained_model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> PreTrainedModel:
    """
    指定された言語の BERT モデルをロードし、ロード済みの BERT モデルを返す。
    一度ロードされていれば、ロード済みの BERT モデルを即座に返す。
    ライブラリ利用時は常に必ず pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある。
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき。
    cache_dir と revision は pretrain_model_name_or_path がリポジトリ名の場合のみ有効。

    Style-Bert-VITS2 v3.0.0 以降は日本語のみサポート:
    - 日本語: ku-nlp/deberta-v2-large-japanese-char-wwm

    Args:
        language (Languages): ロードする学習済みモデルの対象言語 (JP のみサポート)
        pretrained_model_name_or_path (Optional[str]): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)
        cache_dir (Optional[str]): モデルのキャッシュディレクトリ。指定しない場合はデフォルトのキャッシュディレクトリが利用される (デフォルト: None)
        revision (str): モデルの Hugging Face 上の Git リビジョン。指定しない場合は最新の main ブランチの内容が利用される (デフォルト: None)

    Returns:
        PreTrainedModel: ロード済みの BERT モデル
    """

    if language != Languages.JP:
        raise ValueError(
            f"Language {language} not supported. Only JP is supported in v3.0+"
        )

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_models:
        return __loaded_models[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = str(DEFAULT_BERT_MODEL_PATHS[language])

    # BERT モデルをロードし、辞書に格納して返す
    start_time = time.time()
    __loaded_models[language] = AutoModelForMaskedLM.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
    )
    logger.info(
        f"Loaded the {language.name} BERT model from {pretrained_model_name_or_path} ({time.time() - start_time:.2f}s)"
    )

    # 初期ロード時の dtype を記録 (デフォルトは FP32)
    __model_dtypes[language] = torch.float32

    return __loaded_models[language]


def load_tokenizer(
    language: Languages,
    pretrained_model_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """
    指定された言語の BERT トークナイザーをロードし、ロード済みの BERT トークナイザーを返す。
    一度ロードされていれば、ロード済みの BERT トークナイザーを即座に返す。
    ライブラリ利用時は常に必ず pretrain_model_name_or_path (Hugging Face のリポジトリ名 or ローカルのファイルパス) を指定する必要がある。
    ロードにはそれなりに時間がかかるため、ライブラリ利用前に明示的に pretrained_model_name_or_path を指定してロードしておくべき。
    cache_dir と revision は pretrain_model_name_or_path がリポジトリ名の場合のみ有効。

    Style-Bert-VITS2 v3.0.0 以降は日本語のみサポート:
    - 日本語: ku-nlp/deberta-v2-large-japanese-char-wwm

    Args:
        language (Languages): ロードする学習済みモデルの対象言語 (JP のみサポート)
        pretrained_model_name_or_path (Optional[str]): ロードする学習済みモデルの名前またはパス。指定しない場合はデフォルトのパスが利用される (デフォルト: None)
        cache_dir (Optional[str]): モデルのキャッシュディレクトリ。指定しない場合はデフォルトのキャッシュディレクトリが利用される (デフォルト: None)
        revision (str): モデルの Hugging Face 上の Git リビジョン。指定しない場合は最新の main ブランチの内容が利用される (デフォルト: None)

    Returns:
        Union[PreTrainedTokenizer, PreTrainedTokenizerFast]: ロード済みの BERT トークナイザー
    """

    if language != Languages.JP:
        raise ValueError(
            f"Language {language} not supported. Only JP is supported in v3.0+"
        )

    # すでにロード済みの場合はそのまま返す
    if language in __loaded_tokenizers:
        return __loaded_tokenizers[language]

    # pretrained_model_name_or_path が指定されていない場合はデフォルトのパスを利用
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = str(DEFAULT_BERT_MODEL_PATHS[language])

    # BERT トークナイザーをロードし、辞書に格納して返す
    __loaded_tokenizers[language] = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
        use_fast=True,  # デフォルトで True だが念のため明示的に指定
    )
    logger.info(
        f"Loaded the {language.name} BERT tokenizer from {pretrained_model_name_or_path}"
    )

    return __loaded_tokenizers[language]


def transfer_model(
    language: Languages, device: str, dtype: Optional[torch.dtype] = None
) -> None:
    """
    指定された言語の BERT モデルを、指定されたデバイスに移動する。
    モデルのロード後に推論デバイスを変更したい場合に利用する。
    既に指定されたデバイスにモデルがロードされている場合は何も行われない。

    Args:
        language (Languages): モデルを移動する言語
        device (str): モデルを移動するデバイス
        dtype (Optional[torch.dtype]): モデルの dtype (torch.float16, torch.bfloat16 など). None の場合は変換しない
    """
    if language not in __loaded_models:
        raise ValueError(f"BERT model for {language.name} is not loaded.")

    # 既に指定されたデバイスにモデルがロードされている場合は何もしない
    # ex: current_device="cuda:0", device="cuda" → 何もしない
    # ex: current_device="cuda:0", device="cpu" → モデルを CPU に移動
    current_device = str(__loaded_models[language].device)
    current_dtype = __model_dtypes.get(language, torch.float32)

    # dtype が None の場合は FP32 に変換 (デフォルト)
    target_dtype = dtype if dtype is not None else torch.float32

    if current_device.startswith(device):
        # Device is already correct, but check if we need to convert dtype
        if device != "cpu" and current_dtype != target_dtype:
            __loaded_models[language].to(dtype=target_dtype)  # type: ignore
            __model_dtypes[language] = target_dtype
            logger.info(f"Converted the {language.name} BERT model to {target_dtype}")
        return

    __loaded_models[language].to(device)  # type: ignore
    if device != "cpu":
        __loaded_models[language].to(dtype=target_dtype)  # type: ignore
        __model_dtypes[language] = target_dtype
        logger.info(
            f"Transferred the {language.name} BERT model from {current_device} to {device} and converted to {target_dtype}"
        )
    else:
        logger.info(
            f"Transferred the {language.name} BERT model from {current_device} to {device}"
        )


def is_model_loaded(language: Languages) -> bool:
    """
    指定された言語の BERT モデルがロード済みかどうかを返す。
    """

    return language in __loaded_models


def is_tokenizer_loaded(language: Languages) -> bool:
    """
    指定された言語の BERT トークナイザーがロード済みかどうかを返す。
    """

    return language in __loaded_tokenizers


def unload_model(language: Languages) -> None:
    """
    指定された言語の BERT モデルをアンロードする。

    Args:
        language (Languages): アンロードする BERT モデルの言語
    """
    if language in __loaded_models:
        del __loaded_models[language]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Unloaded the {language.name} BERT model")


def unload_tokenizer(language: Languages) -> None:
    """
    指定された言語の BERT トークナイザーをアンロードする。

    Args:
        language (Languages): アンロードする BERT トークナイザーの言語
    """

    if language in __loaded_tokenizers:
        del __loaded_tokenizers[language]
        gc.collect()
        logger.info(f"Unloaded the {language.name} BERT tokenizer")


def unload_all_models() -> None:
    """
    すべての BERT モデルをアンロードする。
    """

    for language in list(__loaded_models.keys()):
        unload_model(language)
    logger.info("Unloaded all BERT models")


def unload_all_tokenizers() -> None:
    """
    すべての BERT トークナイザーをアンロードする。
    """

    for language in list(__loaded_tokenizers.keys()):
        unload_tokenizer(language)
    logger.info("Unloaded all BERT tokenizers")
