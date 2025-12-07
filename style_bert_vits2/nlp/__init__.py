from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

import torch
from numpy.typing import NDArray

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp.symbols import (
    LANGUAGE_ID_MAP,
    LANGUAGE_TONE_START_MAP,
    SYMBOLS,
)


def _build_symbol_to_id(symbols: list[str]) -> dict[str, int]:
    """Build symbol to ID mapping from a list of symbols."""
    return {s: i for i, s in enumerate(symbols)}


# Default symbol mapping (v3)
__symbol_to_id_v3 = _build_symbol_to_id(SYMBOLS)


def extract_bert_feature(
    text: str,
    word2ph: list[int],
    language: Languages,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    dtype: Optional[torch.dtype] = None,
    sep_text: Optional[list[str]] = None,
) -> torch.Tensor:
    """
    テキストから BERT の特徴量を抽出する (PyTorch 推論)

    Args:
        text (str): テキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        language (Languages): テキストの言語 (JP のみサポート)
        device (str): 推論に利用するデバイス
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)
        dtype (Optional[torch.dtype], optional): モデルの dtype (torch.float16, torch.bfloat16 など). None の場合は FP32
        sep_text (Optional[list[str]], optional): 単語単位の単語のリスト (デフォルト: None)

    Returns:
        torch.Tensor: BERT の特徴量
    """

    if language != Languages.JP:
        raise ValueError(
            f"Language {language} not supported. Only JP is supported in v3.0+"
        )

    from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature

    return extract_bert_feature(
        text, word2ph, device, assist_text, assist_text_weight, dtype, sep_text
    )


def extract_bert_feature_onnx(
    text: str,
    word2ph: list[int],
    language: Languages,
    onnx_providers: Sequence[Union[str, tuple[str, dict[str, Any]]]],
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    sep_text: Optional[list[str]] = None,
) -> NDArray[Any]:
    """
    テキストから BERT の特徴量を抽出する (ONNX 推論)

    Args:
        text (str): テキスト
        word2ph (list[int]): 元のテキストの各文字に音素が何個割り当てられるかを表すリスト
        language (Languages): テキストの言語 (JP のみサポート)
        onnx_providers (list[str]): ONNX 推論で利用する ExecutionProvider (CPUExecutionProvider, CUDAExecutionProvider など)
        assist_text (Optional[str], optional): 補助テキスト (デフォルト: None)
        assist_text_weight (float, optional): 補助テキストの重み (デフォルト: 0.7)
        sep_text (Optional[list[str]], optional): 単語単位の単語のリスト (デフォルト: None)

    Returns:
        NDArray[Any]: BERT の特徴量
    """

    if language != Languages.JP:
        raise ValueError(
            f"Language {language} not supported. Only JP is supported in v3.0+"
        )

    from style_bert_vits2.nlp.japanese.bert_feature import extract_bert_feature_onnx

    return extract_bert_feature_onnx(
        text, word2ph, onnx_providers, assist_text, assist_text_weight, sep_text
    )


def _clean_text(
    text: str,
    language: Languages,
    raise_yomi_error: bool = False,
) -> tuple[
    str,
    list[str],
    list[int],
    list[int],
    Optional[list[str]],
    Optional[list[str]],
    Optional[list[str]],
]:
    """
    テキストをクリーニングし、音素に変換する
    この関数では実装の都合上 convert_unsupported_phones_for_current_model() を呼び出さないため、
    必ずこの _clean_text() の代わりに clean_text_with_given_phone_tone() を使うこと

    Args:
        text (str): クリーニングするテキスト
        language (Languages): テキストの言語 (JP のみサポート)
        raise_yomi_error (bool, optional): False の場合、読めない文字が消えたような扱いとして処理される。Defaults to False.

    Returns:
        tuple[str, list[str], list[int], list[int], list[str] | None, list[str] | None, list[str] | None]:
            - クリーニングされたテキスト
            - 音素
            - アクセント
            - 元のテキストの各文字に音素が何個割り当てられるかのリスト
            - 単語単位の単語のリスト
            - 単語単位の単語のカタカナ読みのリスト
            - 単語単位の単語のカタカナ読みに助詞を追加したリスト
    """

    if language != Languages.JP:
        raise ValueError(
            f"Language {language} not supported. Only JP is supported in v3.0+"
        )

    from style_bert_vits2.nlp.japanese.g2p import g2p
    from style_bert_vits2.nlp.japanese.normalizer import normalize_text

    norm_text = normalize_text(text)
    phones, tones, word2ph, sep_text, sep_kata, sep_kata_with_joshi = g2p(
        norm_text,
        raise_yomi_error=raise_yomi_error,
    )

    return norm_text, phones, tones, word2ph, sep_text, sep_kata, sep_kata_with_joshi


def clean_text_with_given_phone_tone(
    text: str,
    language: Languages,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    raise_yomi_error: bool = False,
) -> tuple[
    str,
    list[str],
    list[int],
    list[int],
    Optional[list[str]],
    Optional[list[str]],
    Optional[list[str]],
]:
    """
    テキストをクリーニングし、音素に変換する
    変換時、given_phone や given_tone が与えられた場合はそれを調整して使う
    この関数は内部で convert_unsupported_phones_for_current_model() を自動的に呼び出し、対応していない音素をフォールバックする

    Args:
        text (str): クリーニングするテキスト
        language (Languages): テキストの言語 (JP のみサポート)
        given_phone (Optional[list[str]], optional): 読み上げテキストの読みを表す音素列。指定する場合は given_tone も別途指定が必要. Defaults to None.
        given_tone (Optional[list[int]], optional): アクセントのトーンのリスト. Defaults to None.
        raise_yomi_error (bool, optional): False の場合、読めない文字が消えたような扱いとして処理される。Defaults to False.

    Returns:
        tuple[str, list[str], list[int], list[int], list[str] | None, list[str] | None, list[str] | None]:
            - クリーニングされたテキスト
            - 音素
            - アクセント
            - 元のテキストの各文字に音素が何個割り当てられるかのリスト
            - 単語単位の単語のリスト
            - 単語単位の単語のカタカナ読みのリスト
            - 単語単位の単語のカタカナ読みに助詞を追加したリスト
    """

    # 与えられたテキストをクリーニング
    norm_text, phone, tone, word2ph, sep_text, sep_kata, sep_kata_with_joshi = (
        _clean_text(
            text,
            language,
            raise_yomi_error=raise_yomi_error,
        )
    )

    # phone と tone の両方が与えられた場合はそれを使う
    if given_phone is not None and given_tone is not None:
        # 指定された phone と指定された tone 両方の長さが一致していなければならない
        if len(given_phone) != len(given_tone):
            raise InvalidPhoneError(
                f"Length of given_phone ({len(given_phone)}) != length of given_tone ({len(given_tone)})"
            )
        # 与えられた音素数と pyopenjtalk で生成した読みの音素数が一致しない
        if len(given_phone) != sum(word2ph):
            from style_bert_vits2.nlp.japanese.g2p import adjust_word2ph

            # _clean_text() から取得した word2ph を調整結果で上書き
            word2ph = adjust_word2ph(word2ph, phone, given_phone)
            # 上記処理により word2ph の合計が given_phone の長さと一致するはず
            # それでも一致しないとしたら、len(generated_phone) に比べて len(given_phone) があまりに少なすぎて、
            # 各文字ごとに最低 1 以上の音素を割り当てることが不可能だったことを意味する
            # 通常無理やりにでも辻褄を合わせるため発生しないはずだが、どうしても一致しない場合はエラーとする
            if len(given_phone) != sum(word2ph):
                raise InvalidPhoneError(
                    f"Length of given_phone ({len(given_phone)}) != sum of word2ph ({sum(word2ph)})"
                )
        phone = given_phone
        # 生成あるいは指定された phone と指定された tone 両方の長さが一致していなければならない
        if len(phone) != len(given_tone):
            raise InvalidToneError(
                f"Length of phone ({len(phone)}) != length of given_tone ({len(given_tone)})"
            )
        tone = given_tone

    # tone だけが与えられた場合は _clean_text() で生成した phone と合わせて使う
    elif given_tone is not None:
        # 生成した phone と指定された tone 両方の長さが一致していなければならない
        if len(phone) != len(given_tone):
            raise InvalidToneError(
                f"Length of phone ({len(phone)}) != length of given_tone ({len(given_tone)})"
            )
        tone = given_tone

    # g2p 処理では対応しているが現行モデルでは対応していない特定音素を変換 (フォールバック)
    # この処理は given_phone / given_tone が調整された後に実行する必要がある
    convert_unsupported_phones_for_current_model(phone, tone, word2ph, language)

    return norm_text, phone, tone, word2ph, sep_text, sep_kata, sep_kata_with_joshi


def convert_unsupported_phones_for_current_model(
    phone: list[str],
    tone: list[int],
    word2ph: list[int],
    language: Languages,
) -> None:
    """
    g2p 処理では対応しているが現行モデルでは対応していない特定音素を、対応する音素にフォールバックする
    変更は引数で与えられた phone / tone / word2ph に in-place で適用される
    必ず phone / tone を cleaned_text_to_sequence() に渡す前に、一度だけ実行する必要がある

    Args:
        phone (list[str]): 音素リスト
        tone (list[int]): アクセントリスト
        word2ph (list[int]): 各文字に割り当てられた音素数のリスト
        language (Languages): 言語 (JP のみサポート)
    """

    # ここでは必ず音素数・アクセント数・word2ph の長さが一致するはず（事前チェックとして念のため）
    # 通常起こり得ないが、万が一一致しない場合、誤った対応関係で学習される可能性があるためデータセットに含めるべきでない
    assert len(phone) == len(tone) == sum(word2ph)

    # JP のみサポート
    if language != Languages.JP:
        return

    # 音素変換マップ
    # pyopenjtalk が出力するが JP_SYMBOLS に存在しない音素を、対応する音素に変換する
    PHONE_CONVERSION_MAP = {
        "kw": ("k", "u", "w"),  # 「クヮ」→「クワ」
        "gw": ("g", "u", "w"),  # 「グヮ」→「グワ」
        "fy": ("hy",),  # 「フュ」→「ヒュ」
    }

    # 変換が必要な音素のインデックスを収集
    conversion_indices: list[tuple[int, str]] = []
    for i, p in enumerate(phone):
        if p in PHONE_CONVERSION_MAP:
            conversion_indices.append((i, p))

    # 音素変換が必要な場合のみ処理を実行
    if conversion_indices:
        # インデックスは後ろから処理することで、
        # 前の変換による位置ずれの影響を受けないようにする
        for orig_idx, orig_phone in reversed(conversion_indices):
            # 変換後の音素を取得
            converted_phones = PHONE_CONVERSION_MAP[orig_phone]

            # phone リストの更新
            ## スライスで置換すると要素数が変化する
            phone[orig_idx : orig_idx + 1] = list(converted_phones)

            # tone リストの更新
            ## 元の音素のトーンを、変換後の音素全てに適用
            orig_tone = tone[orig_idx]
            tone[orig_idx : orig_idx + 1] = [orig_tone] * len(converted_phones)

            # word2ph リストの更新
            ## 元の音素が属していた文字のインデックスを特定
            char_idx = 0
            phone_count = 0
            for i, count in enumerate(word2ph):
                if phone_count + count > orig_idx:
                    char_idx = i
                    break
                phone_count += count

            # 該当する文字の音素数を更新
            ## kw, gw の場合、1つの音素が3つの音素に変換されるので、2つ増える
            word2ph[char_idx] += len(converted_phones) - 1

    # ここでは必ず音素数・アクセント数・word2ph の長さが一致するはず
    assert len(phone) == len(tone) == sum(word2ph)


def cleaned_text_to_sequence(
    cleaned_phones: list[str],
    tones: list[int],
    language: Languages,
    symbols: Optional[list[str]] = None,
    tone_start: Optional[int] = None,
    language_id: Optional[int] = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    音素リスト・アクセントリスト・言語を、テキスト内の対応する ID に変換する

    Args:
        cleaned_phones (list[str]): clean_text() でクリーニングされた音素のリスト
        tones (list[int]): 各音素のアクセント
        language (Languages): テキストの言語 (JP のみサポート)
        symbols (Optional[list[str]]): シンボルリスト (None の場合は v3 デフォルト)
        tone_start (Optional[int]): トーンの開始インデックス (None の場合はデフォルト)
        language_id (Optional[int]): 言語 ID (None の場合はデフォルト)

    Returns:
        tuple[list[int], list[int], list[int]]: List of integers corresponding to the symbols in the text
    """

    # Use provided symbols or default to v3
    if symbols is not None:
        symbol_to_id = _build_symbol_to_id(symbols)
    else:
        symbol_to_id = __symbol_to_id_v3

    phones = [symbol_to_id[symbol] for symbol in cleaned_phones]

    # Use provided tone_start or default
    if tone_start is None:
        tone_start = LANGUAGE_TONE_START_MAP[language]
    tones = [i + tone_start for i in tones]

    # Use provided language_id or default
    if language_id is None:
        language_id = LANGUAGE_ID_MAP[language]
    lang_ids = [language_id for _ in phones]

    return phones, tones, lang_ids


class InvalidPhoneError(ValueError):
    pass


class InvalidToneError(ValueError):
    pass
