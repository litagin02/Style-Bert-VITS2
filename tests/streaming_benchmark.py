#!/usr/bin/env python3
"""
Usage: .venv/Scripts/python -m tests.streaming_benchmark [--device cuda] [--model amitaro] [--runs 3]

このスクリプトは infer.py に実装されている infer() 関数と infer_stream() 関数のパフォーマンスを比較し、
https://qiita.com/__dAi00/items/970f0fe66286510537dd の結果と同様の測定を行う。

測定項目:
- 初回チャンク生成までの時間（ストリーミング版のレイテンシ）
- 全音声生成完了までの時間（総処理時間）
- 生成音声の長さごとの効果の変化
"""

import argparse
import time
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from style_bert_vits2.constants import (
    DEFAULT_LENGTH,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    Languages,
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models.infer import infer, infer_stream
from style_bert_vits2.tts_model import TTSModel, TTSModelHolder

from .utils import save_benchmark_audio, set_random_seeds


# 測定用サンプルテキスト
BENCHMARK_TEXTS = [
    {
        # 初回ロード用（ダミー）
        "text": "あああ",
        "description": "Dummy (skip)",
    },
    {
        "text": "こんにちは",
        "description": "短文",
    },
    {
        "text": "東京特許許可局",
        "description": "短文",
    },
    {
        "text": "あのイーハトーヴォのすきとおった風",
        "description": "中文",
    },
    {
        "text": "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら",
        "description": "中文",
    },
    {
        "text": "あのイーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。",
        "description": "長文",
    },
    {
        # From long_inference_benchmark.py - Medium (ROUDOKU)
        "text": "イーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。またそのなかでいっしょになったたくさんのひとたち、ファゼーロとロザーロ、羊飼のミーロや、顔の赤いこどもたち、地主のテーモ、山猫博士のボーガント・デストゥパーゴなど、いまこの暗い巨きな家にはたったひとりがいません。",
        "description": "長文 (ROUDOKU)",
    },
    {
        # From long_inference_benchmark.py - Medium (NEWS)
        "text": "小笠原近海で台風５号が発生しました。今後、北上し、関東から東北の太平洋側に沿って北上した後、北海道付近に到達する可能性が大きくなっています。もし関東へ上陸すれば６年ぶり、東北に上陸すれば２年連続、北海道へ上陸すれば９年ぶりとなります。この台風の進路の特徴とともに、詳しくみていきましょう。",
        "description": "長文 (NEWS)",
    },
    {
        # From long_inference_benchmark.py - Medium Long (ROUDOKU)
        "text": "濁流は、メロスの叫びをせせら笑う如く、ますます激しく躍り狂う。浪は浪を呑み、捲き、煽り立て、そうして時は、刻一刻と消えて行く。今はメロスも覚悟した。泳ぎ切るより他に無い。ああ、神々も照覧あれ！　濁流にも負けぬ愛と誠の偉大な力を、いまこそ発揮して見せる。メロスは、ざんぶと流れに飛び込み、百匹の大蛇のようにのた打ち荒れ狂う浪を相手に、必死の闘争を開始した。満身の力を腕にこめて、押し寄せ渦巻き引きずる流れを、なんのこれしきと掻きわけ掻きわけ、めくらめっぽう獅子奮迅の人の子の姿には、神も哀れと思ったか、ついに憐愍を垂れてくれた。",
        "description": "超長文 (ROUDOKU)",
    },
    {
        # From long_inference_benchmark.py - Long (ROUDOKU) - Melos story
        "text": "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。きょう未明メロスは村を出発し、野を越え山越え、十里はなれた此のシラクスの市にやって来た。メロスには父も、母も無い。女房も無い。十六の、内気な妹と二人暮しだ。この妹は、村の或る律気な一牧人を、近々、花婿として迎える事になっていた。結婚式も間近かなのである。メロスは、それゆえ、花嫁の衣裳やら祝宴の御馳走やらを買いに、はるばる市にやって来たのだ。先ず、その品々を買い集め、それから都の大路をぶらぶら歩いた。メロスには竹馬の友があった。セリヌンティウスである。今は此のシラクスの市で、石工をしている。その友を、これから訪ねてみるつもりなのだ。久しく逢わなかったのだから、訪ねて行くのが楽しみである。歩いているうちにメロスは、まちの様子を怪しく思った。ひっそりしている。もう既に日も落ちて、まちの暗いのは当りまえだが、けれども、なんだか、夜のせいばかりでは無く、市全体が、やけに寂しい。",
        "description": "超長文 (走れメロス)",
    },
]


def measure_infer_performance(
    model: TTSModel,
    text: str,
) -> tuple[float, float, NDArray[np.float32]]:
    """
    通常の infer() 関数のパフォーマンスを測定する。

    Returns:
        tuple: (総処理時間, 生成音声長(秒), 音声データ)
    """
    # Clear CUDA cache before measurement
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.perf_counter()

    net_g = model.net_g
    assert net_g is not None

    style_vec = model.get_style_vector(0, 1.0)  # デフォルトスタイル

    # 比較のために低レベル API で推論を実行
    with torch.inference_mode():
        audio_data = infer(
            text=text,
            style_vec=style_vec,
            sdp_ratio=DEFAULT_SDP_RATIO,
            noise_scale=DEFAULT_NOISE,
            noise_scale_w=DEFAULT_NOISEW,
            length_scale=DEFAULT_LENGTH,
            sid=0,
            language=Languages.JP,
            hps=model.hyper_parameters,
            net_g=net_g,
            device=model.device,
        )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # 生成音声の長さを計算
        audio_duration = len(audio_data) / model.hyper_parameters.data.sampling_rate

        # Clear CUDA cache after measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return total_time, audio_duration, audio_data


def measure_infer_stream_performance(
    model: TTSModel,
    text: str,
) -> tuple[float, float, float, int, list[NDArray[np.float32]]]:
    """
    ストリーミング版 infer_stream() 関数のパフォーマンスを測定する。

    Returns:
        tuple: (初回チャンク時間, 総処理時間, 生成音声長(秒), チャンク数, 音声チャンクリスト)
    """
    # Clear CUDA cache before measurement
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 低レベル API でストリーミング推論を実行
    net_g = model.net_g
    assert net_g is not None

    style_vec = model.get_style_vector(0, 1.0)  # デフォルトスタイル

    start_time = time.perf_counter()
    first_chunk_time = None
    audio_chunks = []
    chunk_count = 0

    # ストリーミング推論を実行
    with torch.inference_mode():
        audio_generator = infer_stream(
            text=text,
            style_vec=style_vec,
            sdp_ratio=DEFAULT_SDP_RATIO,
            noise_scale=DEFAULT_NOISE,
            noise_scale_w=DEFAULT_NOISEW,
            length_scale=DEFAULT_LENGTH,
            sid=0,
            language=Languages.JP,
            hps=model.hyper_parameters,
            net_g=net_g,
            device=model.device,
        )

        for i, audio_chunk in enumerate(audio_generator):
            if i == 0:
                # 初回チャンクが生成された時刻を記録
                first_chunk_time = time.perf_counter() - start_time
            audio_chunks.append(audio_chunk)
            chunk_count += 1

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # 全音声チャンクを結合
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            audio_duration = len(full_audio) / model.hyper_parameters.data.sampling_rate
        else:
            audio_duration = 0.0

        # Clear CUDA cache after measurement
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (
            first_chunk_time or 0.0,
            total_time,
            audio_duration,
            chunk_count,
            audio_chunks,
        )


def run_benchmark(
    device: str = "cpu",
    model_name: str = "amitaro",
    num_runs: int = 3,
    fix_seed: bool = False,
) -> None:
    """
    ベンチマークを実行する。
    """
    from pathlib import Path

    # ランダムシード固定
    if fix_seed:
        set_random_seeds()

    print("=" * 80)
    print("Style-Bert-VITS2 ストリーミング推論パフォーマンス測定")
    print("=" * 80)
    print(f"デバイス: {device}")
    print(f"モデル: {model_name}")
    print(f"測定回数: {num_runs}")
    print(f"ランダムシード固定: {'有効' if fix_seed else '無効'}")
    print("=" * 80)

    # モデルホルダーを初期化
    model_root = Path("model_assets")
    model_holder = TTSModelHolder(
        model_root,
        device,
        onnx_providers=None,
    )
    if len(model_holder.models_info) == 0:
        print("エラー: 音声合成モデルが見つかりませんでした。")
        return

    # 指定されたモデルを検索
    model_info = None
    for info in model_holder.models_info:
        if info.name == model_name:
            model_info = info
            break

    if model_info is None:
        print(f'エラー: モデル "{model_name}" が見つかりませんでした。')
        print("利用可能なモデル:")
        for info in model_holder.models_info:
            print(f"  - {info.name}")
        return

    # Safetensors 形式のモデルファイルを検索
    model_files = [
        f
        for f in model_info.files
        if f.endswith(".safetensors") and not f.startswith(".")
    ]
    if len(model_files) == 0:
        print(
            f'エラー: モデル "{model_name}" の .safetensors ファイルが見つかりませんでした。'
        )
        return

    model_file = model_files[0]
    print(f"使用するモデルファイル: {model_file}")
    print()

    # モデルをロード
    model = model_holder.get_model(model_name, model_file)
    model.load()

    # 結果を保存するリスト
    results = []

    # 各テキストでベンチマークを実行
    for i, test_case in enumerate(BENCHMARK_TEXTS):
        text = cast(str, test_case["text"])
        description = test_case["description"]

        print(f"\n測定中: {description}")
        print(f"テキスト: {text[:50]}..." if len(text) > 50 else f"テキスト: {text}")

        # 複数回実行して平均を取る
        infer_times = []
        infer_durations = []
        stream_first_times = []
        stream_total_times = []
        stream_durations = []
        stream_chunk_counts = []

        # 最後の実行の音声データを保存用に記録
        last_normal_audio = None
        last_stream_audio = None
        last_sample_rate = None

        normal_oom = False
        stream_oom = False

        for run in range(num_runs):
            infer_time = None
            infer_duration = None
            stream_first_time = None
            stream_total_time = None
            stream_duration = None
            chunk_count = None

            # 通常の infer() を測定 (OOMをキャッチして続行)
            if not normal_oom:
                try:
                    infer_time, infer_duration, normal_audio = (
                        measure_infer_performance(
                            model,
                            text,
                        )
                    )
                    infer_times.append(infer_time)
                    infer_durations.append(infer_duration)

                    # 最後の実行の音声データを保存
                    if run == num_runs - 1:
                        last_normal_audio = normal_audio
                        last_sample_rate = model.hyper_parameters.data.sampling_rate

                except torch.cuda.OutOfMemoryError:
                    print(f"  Run {run + 1}: 通常=OOM!")
                    normal_oom = True
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                except Exception as ex:
                    logger.error(f"通常推論でエラー: {ex}")
                    normal_oom = True

            # ストリーミング版 infer_stream() を測定 (OOMをキャッチして続行)
            if not stream_oom:
                try:
                    (
                        stream_first_time,
                        stream_total_time,
                        stream_duration,
                        chunk_count,
                        stream_chunks,
                    ) = measure_infer_stream_performance(
                        model,
                        text,
                    )
                    stream_first_times.append(stream_first_time)
                    stream_total_times.append(stream_total_time)
                    stream_durations.append(stream_duration)
                    stream_chunk_counts.append(chunk_count)

                    # 最後の実行のストリーミング音声データを保存
                    if run == num_runs - 1 and stream_chunks:
                        last_stream_audio = np.concatenate(stream_chunks)

                except torch.cuda.OutOfMemoryError:
                    print(f"  Run {run + 1}: Stream=OOM!")
                    stream_oom = True
                    # Clear CUDA cache
                    torch.cuda.empty_cache()
                except Exception as ex:
                    logger.error(f"ストリーミング推論でエラー: {ex}")
                    stream_oom = True

            # 結果を表示
            normal_str = (
                f"通常={infer_time:.3f}s" if infer_time is not None else "通常=OOM"
            )
            stream_str = (
                f"Stream初回={stream_first_time:.3f}s, Stream合計={stream_total_time:.3f}s, Chunks={chunk_count}"
                if stream_first_time is not None
                else "Stream=OOM"
            )
            print(f"  Run {run + 1}: {normal_str}, {stream_str}")

            # 両方OOMなら終了
            if normal_oom and stream_oom:
                break

        if not infer_times and not stream_first_times:
            print("  両方OOMで測定に失敗しました。")
            continue

        # 初回はロードが入るため捨てる
        if i == 0:
            print("  (初回実行はBERTロードを含むためスキップ)")
            continue

        # 音声ファイルを保存（初回のダミーは除く）
        if (
            last_normal_audio is not None
            and last_stream_audio is not None
            and last_sample_rate is not None
        ):
            save_benchmark_audio(
                last_normal_audio,
                last_sample_rate,
                text,
                "streaming_benchmark",
                "normal",
            )
            save_benchmark_audio(
                last_stream_audio,
                last_sample_rate,
                text,
                "streaming_benchmark",
                "streaming",
            )

        # 平均値を計算 (OOMの場合はNone)
        avg_infer_time = np.mean(infer_times) if infer_times else None
        avg_infer_duration = np.mean(infer_durations) if infer_durations else None
        avg_stream_first_time = (
            np.mean(stream_first_times) if stream_first_times else None
        )
        avg_stream_total_time = (
            np.mean(stream_total_times) if stream_total_times else None
        )
        avg_stream_duration = np.mean(stream_durations) if stream_durations else None
        avg_chunk_count = (
            int(np.mean(stream_chunk_counts)) if stream_chunk_counts else None
        )

        # RTF (Real-Time Factor) = inference_time / audio_duration
        # RTF < 1.0 means faster than real-time, RTF > 1.0 means slower
        rtf_normal = (
            avg_infer_time / avg_infer_duration
            if (avg_infer_time and avg_infer_duration and avg_infer_duration > 0)
            else None
        )
        rtf_stream_total = (
            avg_stream_total_time / avg_stream_duration
            if (
                avg_stream_total_time
                and avg_stream_duration
                and avg_stream_duration > 0
            )
            else None
        )
        rtf_stream_first = (
            avg_stream_first_time / avg_stream_duration
            if (
                avg_stream_first_time
                and avg_stream_duration
                and avg_stream_duration > 0
            )
            else None
        )

        # 結果を保存 (OOMの場合はNoneを許容)
        # 音声長はストリーミングの結果から取得 (OOMでも取れる可能性がある)
        actual_duration = (
            avg_stream_duration if avg_stream_duration else avg_infer_duration
        )

        # 効率とオーバーヘッドの計算 (OOMを考慮)
        efficiency_infer = (
            (avg_infer_duration / avg_infer_time)
            if (avg_infer_time and avg_infer_duration and avg_infer_time > 0)
            else None
        )
        efficiency_stream = (
            (avg_stream_duration / avg_stream_total_time)
            if (
                avg_stream_total_time
                and avg_stream_duration
                and avg_stream_total_time > 0
            )
            else None
        )
        latency_improvement = (
            (avg_infer_time - avg_stream_first_time)
            if (avg_infer_time is not None and avg_stream_first_time is not None)
            else None
        )
        overhead = (
            (avg_stream_total_time - avg_infer_time)
            if (avg_stream_total_time is not None and avg_infer_time is not None)
            else None
        )
        overhead_pct = (
            ((avg_stream_total_time - avg_infer_time) / avg_infer_time * 100)
            if (avg_infer_time and avg_stream_total_time and avg_infer_time > 0)
            else None
        )

        result = {
            "text": text,
            "description": description,
            "actual_duration": actual_duration,
            "infer_time": avg_infer_time,
            "stream_first_time": avg_stream_first_time,
            "stream_total_time": avg_stream_total_time,
            "stream_duration": avg_stream_duration,
            "chunk_count": avg_chunk_count,
            "efficiency_infer": efficiency_infer,
            "efficiency_stream": efficiency_stream,
            "latency_improvement": latency_improvement,
            "overhead": overhead,
            "overhead_pct": overhead_pct,
            "rtf_normal": rtf_normal,
            "rtf_stream_total": rtf_stream_total,
            "rtf_stream_first": rtf_stream_first,
            "normal_oom": normal_oom,
            "stream_oom": stream_oom,
        }
        results.append(result)

        # 個別結果を表示
        duration_str = (
            f"音声長: {actual_duration:.2f}秒" if actual_duration else "音声長: N/A"
        )
        print(f"  {duration_str}")
        if avg_infer_time is not None:
            print(f"  通常推論: {avg_infer_time:.3f}秒 (RTF: {rtf_normal:.3f})")
        else:
            print(f"  通常推論: OOM")
        if avg_stream_first_time is not None:
            print(
                f"  Stream初回: {avg_stream_first_time:.3f}秒 (RTF to first: {rtf_stream_first:.3f})"
            )
            print(
                f"  Stream合計: {avg_stream_total_time:.3f}秒 (Chunks: {avg_chunk_count}, RTF: {rtf_stream_total:.3f})"
            )
        else:
            print(f"  Stream: OOM")
        if latency_improvement is not None:
            print(
                f"  レイテンシ改善: {latency_improvement:.3f}秒 ({latency_improvement / avg_infer_time * 100:.1f}%)"
            )
            print(f"  オーバーヘッド: {overhead:.3f}秒 ({overhead_pct:+.1f}%)")
        elif normal_oom and not stream_oom:
            print(f"  ** 通常推論はOOMだがStreamは成功! **")

    # モデルをアンロード
    model.unload()

    # 総合結果を表示
    print("\n" + "=" * 120)
    print("総合結果")
    print("=" * 120)
    print(
        f"{'音声長':>8} {'通常':>8} {'RTF':>6} {'Stream初回':>12} {'Stream合計':>12} {'RTF':>6} {'Chunks':>6} {'レイテンシ改善':>14} {'オーバーヘッド':>14}"
    )
    print("-" * 120)

    for result in results:
        # Format values with OOM handling
        dur_str = (
            f"{result['actual_duration']:>7.2f}s"
            if result["actual_duration"]
            else "    N/A"
        )
        infer_str = (
            f"{result['infer_time']:>7.3f}s" if result["infer_time"] else "    OOM"
        )
        rtf_n_str = (
            f"{result['rtf_normal']:>6.3f}" if result["rtf_normal"] else "   OOM"
        )
        first_str = (
            f"{result['stream_first_time']:>11.3f}s"
            if result["stream_first_time"]
            else "        OOM"
        )
        total_str = (
            f"{result['stream_total_time']:>11.3f}s"
            if result["stream_total_time"]
            else "        OOM"
        )
        rtf_s_str = (
            f"{result['rtf_stream_total']:>6.3f}"
            if result["rtf_stream_total"]
            else "   OOM"
        )
        chunk_str = f"{result['chunk_count']:>6}" if result["chunk_count"] else "   OOM"

        if result["latency_improvement"] is not None:
            lat_str = f"{result['latency_improvement']:>+7.3f}s ({result['latency_improvement'] / result['infer_time'] * 100:>+5.1f}%)"
            ovh_str = f"{result['overhead']:>+7.3f}s ({result['overhead_pct']:>+5.1f}%)"
        elif result["normal_oom"] and not result["stream_oom"]:
            lat_str = "   N/A (Normal OOM)"
            ovh_str = "  Stream OK!"
        else:
            lat_str = "   N/A"
            ovh_str = "   N/A"

        print(
            f"{dur_str} "
            f"{infer_str} "
            f"{rtf_n_str} "
            f"{first_str} "
            f"{total_str} "
            f"{rtf_s_str} "
            f"{chunk_str} "
            f"{lat_str:>20} "
            f"{ovh_str:>14}"
        )

    print("=" * 120)
    print("\n分析:")

    # OOMケースをカウント
    normal_oom_count = sum(1 for r in results if r["normal_oom"])
    stream_oom_count = sum(1 for r in results if r["stream_oom"])
    if normal_oom_count > 0 or stream_oom_count > 0:
        print(f"- OOM発生: 通常={normal_oom_count}件, Stream={stream_oom_count}件")
        # Streamが成功した場合を強調
        stream_only_success = sum(
            1 for r in results if r["normal_oom"] and not r["stream_oom"]
        )
        if stream_only_success > 0:
            print(
                f"  ** Streamのみ成功: {stream_only_success}件 (通常推論ではOOMだったがStreamは成功!) **"
            )

    # 2秒以上のケースでの改善効果を確認 (OOMを除く)
    long_audio_results = [
        r
        for r in results
        if r["actual_duration"]
        and r["actual_duration"] >= 2.0
        and r["latency_improvement"] is not None
    ]
    if long_audio_results:
        avg_improvement = np.mean(
            [r["latency_improvement"] for r in long_audio_results]
        )
        avg_overhead = np.mean([r["overhead_pct"] for r in long_audio_results])
        avg_rtf_normal_long = np.mean([r["rtf_normal"] for r in long_audio_results])
        avg_rtf_stream_long = np.mean(
            [r["rtf_stream_total"] for r in long_audio_results]
        )
        print(f"- 2秒以上の音声 (OOM除く):")
        print(f"  - レイテンシ改善: 平均 {avg_improvement:.3f}秒")
        print(f"  - オーバーヘッド: 平均 {avg_overhead:+.1f}%")
        print(f"  - RTF (通常): 平均 {avg_rtf_normal_long:.3f}")
        print(f"  - RTF (Stream): 平均 {avg_rtf_stream_long:.3f}")

    # 全体の効率比較 (OOMを除く)
    valid_results = [r for r in results if r["efficiency_infer"] is not None]
    stream_valid_results = [r for r in results if r["efficiency_stream"] is not None]

    if valid_results:
        avg_infer_efficiency = np.mean([r["efficiency_infer"] for r in valid_results])
        avg_rtf_normal = np.mean([r["rtf_normal"] for r in valid_results])
        print(f"- 通常推論 (OOM除く {len(valid_results)}件):")
        print(f"  - 推論効率: {avg_infer_efficiency:.2f}x (音声長/推論時間)")
        print(f"  - RTF: 平均 {avg_rtf_normal:.3f} (< 1.0 = faster than real-time)")

    if stream_valid_results:
        avg_stream_efficiency = np.mean(
            [r["efficiency_stream"] for r in stream_valid_results]
        )
        avg_rtf_stream = np.mean([r["rtf_stream_total"] for r in stream_valid_results])
        print(f"- ストリーミング (OOM除く {len(stream_valid_results)}件):")
        print(f"  - 推論効率: {avg_stream_efficiency:.2f}x (音声長/推論時間)")
        print(f"  - RTF: 平均 {avg_rtf_stream:.3f} (< 1.0 = faster than real-time)")

    # オーバーヘッドの分析 (両方成功したケース)
    both_valid = [r for r in results if r["overhead_pct"] is not None]
    if both_valid:
        avg_overhead = np.mean([r["overhead_pct"] for r in both_valid])
        print(
            f"- 平均オーバーヘッド (両方成功 {len(both_valid)}件): {avg_overhead:+.1f}%"
        )

    print("=" * 120)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Style-Bert-VITS2 ストリーミング推論パフォーマンス測定"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="推論に使用するデバイス (default: cuda)",
    )
    parser.add_argument(
        "--model",
        default="amitaro",
        help="使用するモデル名 (default: amitaro)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="各テストケースの実行回数 (default: 3)",
    )
    parser.add_argument(
        "--fix-seed",
        action="store_true",
        help="ランダムシードを固定して再現性を確保する",
    )

    args = parser.parse_args()

    try:
        run_benchmark(
            device=args.device,
            model_name=args.model,
            num_runs=args.runs,
            fix_seed=args.fix_seed,
        )
    except KeyboardInterrupt:
        print("\nベンチマークが中断されました。")
    except Exception as ex:
        logger.exception(f"ベンチマーク実行中にエラーが発生しました: {ex}")


if __name__ == "__main__":
    main()
