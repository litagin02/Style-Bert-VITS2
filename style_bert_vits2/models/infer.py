from collections.abc import Iterator
from typing import Any, Optional, cast

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors import safe_open

from style_bert_vits2.constants import Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons, utils
from style_bert_vits2.models.hyper_parameters import HyperParameters
from style_bert_vits2.models.models import SynthesizerTrn
from style_bert_vits2.nlp import (
    clean_text_with_given_phone_tone,
    cleaned_text_to_sequence,
    extract_bert_feature,
)
from style_bert_vits2.nlp.symbols import (
    LANGUAGE_ID_MAP,
    LANGUAGE_ID_MAP_V2,
    LANGUAGE_TONE_START_MAP,
    LANGUAGE_TONE_START_MAP_V2,
    NUM_TONES_V2,
    NUM_TONES_V3,
    SYMBOLS,
    SYMBOLS_V2,
    SYMBOLS_V3,
)


def _detect_vocab_size(model_path: str) -> int:
    """
    safetensors ファイルから語彙サイズを検出する。

    Args:
        model_path: モデルファイルのパス

    Returns:
        int: 検出された語彙サイズ
    """
    if not model_path.endswith(".safetensors"):
        # .pth/.pt ファイルの場合は v2 (legacy) と仮定
        return len(SYMBOLS_V2)

    with safe_open(model_path, framework="pt") as f:
        if "enc_p.emb.weight" in f.keys():
            emb_shape = f.get_tensor("enc_p.emb.weight").shape
            return emb_shape[0]

    # フォールバック
    return len(SYMBOLS_V2)


def _get_model_params_for_version(
    model_path: str,
) -> tuple[list[str], int, int, int, int]:
    """
    モデルファイルに適した SYMBOLS リスト、トーン数、言語数、JP用トーン開始位置、JP用言語IDを返す。

    Args:
        model_path: モデルファイルのパス

    Returns:
        tuple: (symbols, n_tones, n_languages, jp_tone_start, jp_language_id)
    """
    vocab_size = _detect_vocab_size(model_path)

    if vocab_size == len(SYMBOLS_V2):
        logger.info(f"Detected legacy v2 model (vocab_size={vocab_size})")
        # v2: 12 tones (ZH 0-5, JP 6-7, EN 8-11), 3 languages (ZH, JP, EN)
        jp_tone_start = LANGUAGE_TONE_START_MAP_V2["JP"]  # 6
        jp_language_id = LANGUAGE_ID_MAP_V2["JP"]  # 1
        return SYMBOLS_V2, NUM_TONES_V2, 3, jp_tone_start, jp_language_id
    elif vocab_size == len(SYMBOLS_V3):
        logger.info(f"Detected v3 model (vocab_size={vocab_size})")
        # v3: 2 tones (JP only), 1 language (JP only)
        jp_tone_start = LANGUAGE_TONE_START_MAP["JP"]  # 0
        jp_language_id = LANGUAGE_ID_MAP["JP"]  # 0
        return SYMBOLS_V3, NUM_TONES_V3, 1, jp_tone_start, jp_language_id
    else:
        # 未知の語彙サイズの場合は v2 にフォールバック
        logger.warning(f"Unknown vocab size {vocab_size}, falling back to v2 symbols")
        jp_tone_start = LANGUAGE_TONE_START_MAP_V2["JP"]
        jp_language_id = LANGUAGE_ID_MAP_V2["JP"]
        return SYMBOLS_V2, NUM_TONES_V2, 3, jp_tone_start, jp_language_id


def get_net_g(
    model_path: str,
    version: str,
    device: str,
    hps: HyperParameters,
    model_dtype: Optional[torch.dtype] = None,
) -> SynthesizerTrn:
    """
    モデルをロードして返す。

    v3.0.0 以降は JP-Extra モデルのみサポート。
    pre-3.x モデルとの後方互換性のため、語彙サイズを自動検出する。
    """
    if not version.endswith("JP-Extra"):
        raise ValueError(
            f"Only JP-Extra models are supported in v3.0+. Got version: {version}"
        )

    # 語彙サイズ、トーン数、言語数、JP用マッピングを検出
    symbols, n_tones, n_languages, jp_tone_start, jp_language_id = (
        _get_model_params_for_version(model_path)
    )
    vocab_size = len(symbols)

    logger.info(
        f"Using JP-Extra model with vocab_size={vocab_size}, n_tones={n_tones}, n_languages={n_languages}"
    )

    # use meta device to speed up instantiation
    with torch.device("meta"):
        net_g = SynthesizerTrn(
            n_vocab=vocab_size,
            spec_channels=hps.data.filter_length // 2 + 1,
            segment_size=hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            # Tone and language counts (different for v2 vs v3 models)
            n_tones=n_tones,
            n_languages=n_languages,
            # hps.model 以下のすべての値を引数に渡す
            use_spk_conditioned_encoder=hps.model.use_spk_conditioned_encoder,
            use_noise_scaled_mas=hps.model.use_noise_scaled_mas,
            use_mel_posterior_encoder=hps.model.use_mel_posterior_encoder,
            use_duration_discriminator=hps.model.use_duration_discriminator,
            use_wavlm_discriminator=hps.model.use_wavlm_discriminator,
            inter_channels=hps.model.inter_channels,
            hidden_channels=hps.model.hidden_channels,
            filter_channels=hps.model.filter_channels,
            n_heads=hps.model.n_heads,
            n_layers=hps.model.n_layers,
            kernel_size=hps.model.kernel_size,
            p_dropout=hps.model.p_dropout,
            resblock=hps.model.resblock,
            resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
            resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
            upsample_rates=hps.model.upsample_rates,
            upsample_initial_channel=hps.model.upsample_initial_channel,
            upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
            n_layers_q=hps.model.n_layers_q,
            use_spectral_norm=hps.model.use_spectral_norm,
            gin_channels=hps.model.gin_channels,
            slm=hps.model.slm,
        )

    if model_path.endswith(".pth") or model_path.endswith(".pt"):
        # .pth の場合は assign=True は使えないので、一度 CPU に実体化してからロードして GPU に送る（従来通り）
        # ただし meta からの実体化は to_empty を使う
        net_g = net_g.to_empty(device=device)
        _ = utils.checkpoints.load_checkpoint(
            model_path, net_g, None, skip_optimizer=True, device=device
        )
    elif model_path.endswith(".safetensors"):
        # safetensors の場合は assign=True を使って直接デバイスにロード
        _ = utils.safetensors.load_safetensors(
            model_path, net_g, True, device=device, assign=True
        )
    else:
        raise ValueError(f"Unknown model format: {model_path}")

    _ = net_g.eval()

    # Dtype conversion
    if model_dtype is not None:
        net_g.to(dtype=model_dtype)
        logger.info(f"Entire model converted to {model_dtype}")

    # Generator (Decoder) の推論最適化: weight_norm を取り除く
    # 学習完了後の推論時には weight_norm は不要なオーバーヘッドとなるため除去しておく
    # 精度に影響はなく、単に計算効率が向上する
    net_g.dec.remove_weight_norm()
    logger.info(
        "Generator module weight normalization removed for inference optimization"
    )

    # Store symbols and mappings for later use in get_text
    net_g._symbols = symbols  # type: ignore
    net_g._jp_tone_start = jp_tone_start  # type: ignore
    net_g._jp_language_id = jp_language_id  # type: ignore

    return net_g


def get_text(
    text: str,
    language_str: Languages,
    hps: HyperParameters,
    device: str,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    dtype: Optional[torch.dtype] = None,
    symbols: Optional[list[str]] = None,
    tone_start: Optional[int] = None,
    language_id: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    テキストを音素・トーン・言語 ID・BERT 特徴量に変換する。

    v3.0.0 以降は日本語 (JP) のみサポート。

    Args:
        symbols: モデルに対応するシンボルリスト (None の場合は v3 デフォルト)
        tone_start: JP トーンの開始インデックス (None の場合はデフォルト)
        language_id: JP の言語 ID (None の場合はデフォルト)

    Returns:
        tuple: (ja_bert, phone, tone, language)
    """
    if language_str != Languages.JP:
        raise ValueError(
            f"Language {language_str} not supported. Only JP is supported in v3.0+"
        )

    norm_text, phone, tone, word2ph, sep_text, _, _ = clean_text_with_given_phone_tone(
        text,
        language_str,
        given_phone=given_phone,
        given_tone=given_tone,
        # 推論時のみ呼び出されるので、raise_yomi_error は False に設定
        raise_yomi_error=False,
    )
    phone, tone, language = cleaned_text_to_sequence(
        phone,
        tone,
        language_str,
        symbols=symbols,
        tone_start=tone_start,
        language_id=language_id,
    )

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    ja_bert = extract_bert_feature(
        norm_text,
        word2ph,
        language_str,
        device,
        assist_text,
        assist_text_weight,
        dtype=dtype,
        sep_text=sep_text,
    )
    del word2ph
    assert ja_bert.shape[-1] == len(phone), phone

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return ja_bert, phone, tone, language


def infer(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,
    language: Languages,
    hps: HyperParameters,
    net_g: SynthesizerTrn,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    bert_dtype: Optional[torch.dtype] = None,
) -> NDArray[Any]:
    """
    テキストから音声を生成する。

    v3.0.0 以降は日本語 (JP) のみサポート。
    """
    # Get model-specific symbols and mappings
    symbols = getattr(net_g, "_symbols", None)
    tone_start = getattr(net_g, "_jp_tone_start", None)
    language_id = getattr(net_g, "_jp_language_id", None)

    ja_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
        dtype=bert_dtype,
        symbols=symbols,
        tone_start=tone_start,
        language_id=language_id,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        ja_bert = ja_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        ja_bert = ja_bert[:, :-2]

    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)
        del phones
        sid_tensor = torch.LongTensor([sid]).to(device)

        output = net_g.infer(
            x_tst,
            x_tst_lengths,
            sid_tensor,
            tones,
            lang_ids,
            ja_bert,
            style_vec=style_vec_tensor,
            length_scale=length_scale,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
        )

        audio = output[0][0, 0].data.cpu().float().numpy()

        del (
            x_tst,
            tones,
            lang_ids,
            x_tst_lengths,
            sid_tensor,
            ja_bert,
            style_vec_tensor,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio


def infer_stream(
    text: str,
    style_vec: NDArray[Any],
    sdp_ratio: float,
    noise_scale: float,
    noise_scale_w: float,
    length_scale: float,
    sid: int,
    language: Languages,
    hps: HyperParameters,
    net_g: SynthesizerTrn,
    device: str,
    skip_start: bool = False,
    skip_end: bool = False,
    assist_text: Optional[str] = None,
    assist_text_weight: float = 0.7,
    given_phone: Optional[list[str]] = None,
    given_tone: Optional[list[int]] = None,
    bert_dtype: Optional[torch.dtype] = None,
    chunk_size: int = 100,
    overlap_size: int = 16,
) -> Iterator[NDArray[np.float32]]:
    """
    ストリーミング推論を実行する関数。
    Generator 部分のみストリーミング処理を行い、音声チャンクを逐次 yield する。

    v3.0.0 以降は日本語 (JP) のみサポート。

    Args:
        text: 読み上げるテキスト
        style_vec: スタイルベクトル
        sdp_ratio: SDP と DP の混合比
        noise_scale: ノイズスケール
        noise_scale_w: ノイズスケール W
        length_scale: 長さスケール
        sid: 話者 ID
        language: 言語 (JP のみ)
        hps: ハイパーパラメータ
        net_g: 音声合成モデル
        device: デバイス
        skip_start: 先頭をスキップするか
        skip_end: 末尾をスキップするか
        assist_text: 補助テキスト
        assist_text_weight: 補助テキストの重み
        given_phone: 指定された音素列
        given_tone: 指定されたトーン列
        bert_dtype: BERT の dtype
        chunk_size: チャンクサイズ (フレーム数)
        overlap_size: オーバーラップサイズ (フレーム数)

    Yields:
        NDArray[np.float32]: 音声チャンク

    Reference: https://qiita.com/__dAi00/items/970f0fe66286510537dd
    """
    assert chunk_size > overlap_size, (
        f"chunk_size ({chunk_size}) must be larger than overlap_size ({overlap_size})"
    )
    assert chunk_size > 0 and overlap_size > 0, (
        "chunk_size and overlap_size must be positive"
    )
    assert overlap_size % 2 == 0, (
        "overlap_size must be even for proper margin calculation"
    )

    # Get model-specific symbols and mappings
    symbols = getattr(net_g, "_symbols", None)
    tone_start = getattr(net_g, "_jp_tone_start", None)
    language_id = getattr(net_g, "_jp_language_id", None)

    ja_bert, phones, tones, lang_ids = get_text(
        text,
        language,
        hps,
        device,
        assist_text=assist_text,
        assist_text_weight=assist_text_weight,
        given_phone=given_phone,
        given_tone=given_tone,
        dtype=bert_dtype,
        symbols=symbols,
        tone_start=tone_start,
        language_id=language_id,
    )
    if skip_start:
        phones = phones[3:]
        tones = tones[3:]
        lang_ids = lang_ids[3:]
        ja_bert = ja_bert[:, 3:]
    if skip_end:
        phones = phones[:-2]
        tones = tones[:-2]
        lang_ids = lang_ids[:-2]
        ja_bert = ja_bert[:, :-2]

    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        style_vec_tensor = torch.from_numpy(style_vec).to(device).unsqueeze(0)
        del phones
        sid_tensor = torch.LongTensor([sid]).to(device)

        # Generator への入力特徴量を生成
        z, y_mask, g, attn, z_p, m_p, logs_p = net_g.infer_input_feature(
            x_tst,
            x_tst_lengths,
            sid_tensor,
            tones,
            lang_ids,
            ja_bert,
            style_vec=style_vec_tensor,
            length_scale=length_scale,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
        )

        # Generator 部分のストリーミング処理
        z_input = z * y_mask
        total_length = z_input.shape[2]

        # 全体のアップサンプリング率を計算
        total_upsample_factor = int(np.prod(hps.model.upsample_rates))
        margin_frames = overlap_size // 2

        # Decoder の dtype を取得し、g を事前に変換 (ループ外で1回だけ)
        dec_dtype = next(net_g.dec.parameters()).dtype
        g_dec = g.to(dec_dtype)
        z_input_dec = z_input.to(dec_dtype)

        for start_idx in range(0, total_length, chunk_size - overlap_size):
            end_idx = min(start_idx + chunk_size, total_length)
            chunk = z_input_dec[:, :, start_idx:end_idx]

            # Decoder を実行 (dtype変換は不要、事前に変換済み)
            chunk_output = net_g.dec(chunk, g=g_dec)

            # オーバーラップ処理
            current_output_length = chunk_output.shape[2]

            trim_left = 0
            if start_idx != 0:
                trim_left = margin_frames * total_upsample_factor

            trim_right = 0
            if end_idx != total_length:
                trim_right = margin_frames * total_upsample_factor

            start_slice = trim_left
            end_slice = current_output_length - trim_right

            if start_slice < end_slice:
                valid_audio = chunk_output[:, :, start_slice:end_slice]
                audio_chunk = valid_audio[0, 0].data.cpu().float().numpy()
                if audio_chunk.size > 0:
                    yield audio_chunk

        del (
            x_tst,
            tones,
            lang_ids,
            x_tst_lengths,
            sid_tensor,
            ja_bert,
            style_vec_tensor,
            z,
            y_mask,
            g,
            g_dec,
            z_input,
            z_input_dec,
            attn,
            z_p,
            m_p,
            logs_p,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
