#!/usr/bin/env python3
"""
Usage: .venv/Scripts/python -m tests.benchmark_dtype [--device cuda] [--model amitaro] [--runs 3]

Style-Bert-VITS2 dtype comparison benchmark script

This script measures inference performance and memory usage for different dtype configurations:
- FP32 (default)
- FP16
- BF16

Results are saved to tests/benchmark_dtype_results.txt
"""

import argparse
import datetime
import gc
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from style_bert_vits2.constants import BASE_DIR, Languages
from style_bert_vits2.logging import logger
from style_bert_vits2.tts_model import TTSModelHolder


# Test texts of varying lengths (from tests/long_inference_benchmark.py)
BENCHMARK_TEXTS = [
    {
        "text": "こんにちは、世界！",
        "description": "Short (9 chars)",
    },
    {
        "text": "イーハトーヴォのすきとおった風、夏でも底に冷たさをもつ青いそら、うつくしい森で飾られたモリーオ市、郊外のぎらぎらひかる草の波。またそのなかでいっしょになったたくさんのひとたち、ファゼーロとロザーロ、羊飼のミーロや、顔の赤いこどもたち、地主のテーモ、山猫博士のボーガント・デストゥパーゴなど、いまこの暗い巨きな家にはたったひとりがいません。",
        "description": "Medium (168 chars)",
    },
    {
        "text": "濁流は、メロスの叫びをせせら笑う如く、ますます激しく躍り狂う。浪は浪を呑み、捲き、煽り立て、そうして時は、刻一刻と消えて行く。今はメロスも覚悟した。泳ぎ切るより他に無い。ああ、神々も照覧あれ！　濁流にも負けぬ愛と誠の偉大な力を、いまこそ発揮して見せる。メロスは、ざんぶと流れに飛び込み、百匹の大蛇のようにのた打ち荒れ狂う浪を相手に、必死の闘争を開始した。満身の力を腕にこめて、押し寄せ渦巻き引きずる流れを、なんのこれしきと掻きわけ掻きわけ、めくらめっぽう獅子奮迅の人の子の姿には、神も哀れと思ったか、ついに憐愍を垂れてくれた。",
        "description": "Medium-Long (272 chars)",
    },
    {
        "text": "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。きょう未明メロスは村を出発し、野を越え山越え、十里はなれた此のシラクスの市にやって来た。メロスには父も、母も無い。女房も無い。十六の、内気な妹と二人暮しだ。この妹は、村の或る律気な一牧人を、近々、花婿として迎える事になっていた。結婚式も間近かなのである。メロスは、それゆえ、花嫁の衣裳やら祝宴の御馳走やらを買いに、はるばる市にやって来たのだ。先ず、その品々を買い集め、それから都の大路をぶらぶら歩いた。メロスには竹馬の友があった。セリヌンティウスである。今は此のシラクスの市で、石工をしている。その友を、これから訪ねてみるつもりなのだ。久しく逢わなかったのだから、訪ねて行くのが楽しみである。歩いているうちにメロスは、まちの様子を怪しく思った。ひっそりしている。もう既に日も落ちて、まちの暗いのは当りまえだが、けれども、なんだか、夜のせいばかりでは無く、市全体が、やけに寂しい。",
        "description": "Long (488 chars)",
    },
    {
        "text": "台風５号は、今後、発達しながら北上し、あす１４日は自動車並みの速度で関東から東北の太平洋側を北上し、あさって１５日には北海道付近に達する予想です。一般に台風に伴う風や雨は、進行方向に向かって、中心の右側ほど強く、左側ほど強くない傾向があります。これは台風の右側は、台風を流す風の向きと中心に吹き込む風の向きが一緒になり、風や雨の勢いが増すためで、左側は台風を流す風の向きと中心に吹き込む風の向きが逆になり、互いに打ち消し合って、右側ほど強くはならないためです。（右側は危険半円、左側は可航半円とも呼ばれます。）今回の台風５号は、関東以北の太平洋側を北上するため、おおむね陸地の上は、台風の進行方向の左側に入ることが予想されます。このため、右側に入るよりは、大荒れの度合いは小さいことになりそうですが、とはいえ、もちろん大雨や強風（暴風）、高波などには十分な警戒が必要です。今回の台風５号がもし関東へ上陸したら、６年前に千葉県に上陸し、甚大な暴風の被害をもたらした台風１５号以来となり、もし北海道へ上陸したら、９年前の台風１１号以来となります。一方、東北へは昨年も上陸していますので、２年連続となります。",
        "description": "Very Long (523 chars)",
    },
]

# Dtype configurations to test
DTYPE_CONFIGS = [
    {"name": "FP32", "model_dtype": None},
    {"name": "FP16", "model_dtype": torch.float16},
    {"name": "BF16", "model_dtype": torch.bfloat16},
]


def measure_inference(
    model_holder: TTSModelHolder,
    model_name: str,
    model_file: str,
    text: str,
    num_runs: int,
    device: str,
) -> tuple[float, float, float, float]:
    """
    Measure inference time and memory usage.

    Returns:
        tuple: (avg_time, min_time, max_time, peak_memory_mb)
    """
    model = model_holder.get_model(model_name, model_file)
    model.load()

    times = []
    peak_memory = 0.0

    # Clear CUDA cache before measurement
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

    # Warmup run (not counted)
    try:
        model.infer(text=text, language=Languages.JP, speaker_id=0)
    except Exception as e:
        model.unload()
        raise e

    # Reset memory stats after warmup
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Actual measurements
    for _ in range(num_runs):
        start_time = time.perf_counter()
        model.infer(text=text, language=Languages.JP, speaker_id=0)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    # Get peak memory usage
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB

    model.unload()

    return np.mean(times), np.min(times), np.max(times), peak_memory


def run_benchmark(
    device: str = "cuda",
    model_name: str = "koharune-ami",
    num_runs: int = 3,
    output_file: str = "benchmark_dtype_results.txt",
) -> None:
    """
    Run dtype comparison benchmark.
    """
    if device == "cpu":
        print(
            "WARNING: Running on CPU. FP16/BF16 may not work properly or show performance benefits."
        )

    results = []
    header_lines = []

    header_lines.append("=" * 100)
    header_lines.append("Style-Bert-VITS2 Dtype Comparison Benchmark")
    header_lines.append("=" * 100)
    header_lines.append(
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    header_lines.append(f"Device: {device}")
    header_lines.append(f"Model: {model_name}")
    header_lines.append(f"Runs per test: {num_runs}")
    header_lines.append(f"PyTorch version: {torch.__version__}")
    if device == "cuda":
        header_lines.append(f"CUDA version: {torch.version.cuda}")
        header_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    header_lines.append("=" * 100)
    header_lines.append("")

    for line in header_lines:
        print(line)

    # Find model file
    model_holder = TTSModelHolder(
        BASE_DIR / "model_assets",
        device,
        onnx_providers=[],
        ignore_onnx=True,
        model_dtype=None,
    )

    model_info = None
    for info in model_holder.models_info:
        if info.name == model_name:
            model_info = info
            break

    if model_info is None:
        print(f"ERROR: Model '{model_name}' not found")
        print("Available models:")
        for info in model_holder.models_info:
            print(f"  - {info.name}")
        return

    model_files = [f for f in model_info.files if f.endswith(".safetensors")]
    if not model_files:
        print(f"ERROR: No .safetensors file found for model '{model_name}'")
        return

    model_file = model_files[0]
    print(f"Using model file: {model_file}")
    print("")

    # Run benchmarks for each dtype configuration
    for dtype_config in DTYPE_CONFIGS:
        dtype_name = dtype_config["name"]
        model_dtype = dtype_config["model_dtype"]

        print(f"Testing: {dtype_name}")
        print("-" * 50)

        # Update model holder settings
        model_holder.model_dtype = model_dtype
        model_holder.current_model = None  # Force reload

        dtype_results = {"name": dtype_name, "texts": []}

        for test_case in BENCHMARK_TEXTS:
            text = test_case["text"]
            description = test_case["description"]

            try:
                avg_time, min_time, max_time, peak_memory = measure_inference(
                    model_holder, model_name, model_file, text, num_runs, device
                )

                result = {
                    "description": description,
                    "text_length": len(text),
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "peak_memory": peak_memory,
                }
                dtype_results["texts"].append(result)

                mem_str = f", mem={peak_memory:.0f}MB" if device == "cuda" else ""
                print(
                    f"  {description}: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s{mem_str}"
                )

            except torch.cuda.OutOfMemoryError as e:
                print(f"  {description}: OOM ERROR - {e}")
                dtype_results["texts"].append(
                    {
                        "description": description,
                        "text_length": len(text),
                        "error": "OOM",
                    }
                )
                # Clear CUDA cache after OOM
                if device == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"  {description}: ERROR - {e}")
                dtype_results["texts"].append(
                    {
                        "description": description,
                        "text_length": len(text),
                        "error": str(e),
                    }
                )

        results.append(dtype_results)
        print("")

    # Generate report
    report_lines = header_lines.copy()
    report_lines.append("DETAILED RESULTS")
    report_lines.append("=" * 100)
    report_lines.append("")

    for dtype_result in results:
        report_lines.append(f"[{dtype_result['name']}]")
        report_lines.append("-" * 50)
        for text_result in dtype_result["texts"]:
            if "error" in text_result:
                report_lines.append(
                    f"  {text_result['description']}: ERROR - {text_result['error']}"
                )
            else:
                mem_str = (
                    f", mem={text_result['peak_memory']:.0f}MB"
                    if text_result.get("peak_memory", 0) > 0
                    else ""
                )
                report_lines.append(
                    f"  {text_result['description']}: "
                    f"avg={text_result['avg_time']:.3f}s, min={text_result['min_time']:.3f}s, max={text_result['max_time']:.3f}s{mem_str}"
                )
        report_lines.append("")

    # Summary table
    report_lines.append("")
    report_lines.append("SUMMARY TABLE (Average inference time in seconds)")
    report_lines.append("=" * 120)

    # Header row
    header = f"{'Dtype':<25}"
    for test_case in BENCHMARK_TEXTS:
        desc_short = test_case["description"].split(" ")[0]
        header += f" {desc_short:>12}"
    header += f" {'Overall Avg':>12}"
    report_lines.append(header)
    report_lines.append("-" * 120)

    # Data rows
    for dtype_result in results:
        row = f"{dtype_result['name']:<25}"
        times = []
        for text_result in dtype_result["texts"]:
            if "error" in text_result:
                row += f" {'ERROR':>12}"
            else:
                row += f" {text_result['avg_time']:>12.3f}"
                times.append(text_result["avg_time"])

        if times:
            row += f" {np.mean(times):>12.3f}"
        else:
            row += f" {'N/A':>12}"
        report_lines.append(row)

    report_lines.append("=" * 120)

    # Speedup comparison (relative to FP32)
    report_lines.append("")
    report_lines.append("SPEEDUP vs FP32 (higher is better)")
    report_lines.append("=" * 120)

    # Find FP32 baseline
    fp32_times = {}
    for dtype_result in results:
        if dtype_result["name"] == "FP32":
            for text_result in dtype_result["texts"]:
                if "error" not in text_result:
                    fp32_times[text_result["description"]] = text_result["avg_time"]
            break

    header = f"{'Dtype':<25}"
    for test_case in BENCHMARK_TEXTS:
        desc_short = test_case["description"].split(" ")[0]  # Short, Medium, Long, etc.
        header += f" {desc_short:>12}"
    header += f" {'Overall':>12}"
    report_lines.append(header)
    report_lines.append("-" * 120)

    for dtype_result in results:
        row = f"{dtype_result['name']:<25}"
        speedups = []
        for text_result in dtype_result["texts"]:
            desc = text_result["description"]
            if "error" in text_result or desc not in fp32_times:
                row += f" {'N/A':>12}"
            else:
                speedup = fp32_times[desc] / text_result["avg_time"]
                speedups.append(speedup)
                row += f" {speedup:>11.2f}x"

        if speedups:
            row += f" {np.mean(speedups):>11.2f}x"
        else:
            row += f" {'N/A':>12}"
        report_lines.append(row)

    report_lines.append("=" * 120)

    # Memory usage table (only for CUDA)
    if device == "cuda":
        report_lines.append("")
        report_lines.append("PEAK MEMORY USAGE (MB)")
        report_lines.append("=" * 120)

        header = f"{'Dtype':<25}"
        for test_case in BENCHMARK_TEXTS:
            desc_short = test_case["description"].split(" ")[0]
            header += f" {desc_short:>12}"
        header += f" {'Max':>12}"
        report_lines.append(header)
        report_lines.append("-" * 120)

        for dtype_result in results:
            row = f"{dtype_result['name']:<25}"
            memories = []
            for text_result in dtype_result["texts"]:
                if "error" in text_result:
                    row += f" {'ERROR':>12}"
                else:
                    mem = text_result.get("peak_memory", 0)
                    memories.append(mem)
                    row += f" {mem:>12.0f}"

            if memories:
                row += f" {max(memories):>12.0f}"
            else:
                row += f" {'N/A':>12}"
            report_lines.append(row)

        report_lines.append("=" * 120)

        # Memory savings vs FP32
        report_lines.append("")
        report_lines.append("MEMORY SAVINGS vs FP32 (lower is better)")
        report_lines.append("=" * 120)

        # Find FP32 baseline memory
        fp32_memory = {}
        for dtype_result in results:
            if dtype_result["name"] == "FP32":
                for text_result in dtype_result["texts"]:
                    if "error" not in text_result:
                        fp32_memory[text_result["description"]] = text_result.get(
                            "peak_memory", 0
                        )
                break

        header = f"{'Dtype':<25}"
        for test_case in BENCHMARK_TEXTS:
            desc_short = test_case["description"].split(" ")[0]
            header += f" {desc_short:>12}"
        header += f" {'Avg':>12}"
        report_lines.append(header)
        report_lines.append("-" * 120)

        for dtype_result in results:
            row = f"{dtype_result['name']:<25}"
            ratios = []
            for text_result in dtype_result["texts"]:
                desc = text_result["description"]
                if (
                    "error" in text_result
                    or desc not in fp32_memory
                    or fp32_memory[desc] == 0
                ):
                    row += f" {'N/A':>12}"
                else:
                    ratio = text_result.get("peak_memory", 0) / fp32_memory[desc]
                    ratios.append(ratio)
                    row += f" {ratio:>11.2f}x"

            if ratios:
                row += f" {np.mean(ratios):>11.2f}x"
            else:
                row += f" {'N/A':>12}"
            report_lines.append(row)

        report_lines.append("=" * 120)

    report_lines.append("")
    report_lines.append("Notes:")
    report_lines.append(
        "- FP16/BF16 converts the model to lower precision for faster inference"
    )
    report_lines.append("- Speedup > 1.0 means faster than FP32")
    report_lines.append("- Memory ratio < 1.0 means less memory than FP32")
    report_lines.append("- OOM = Out of Memory error")
    report_lines.append("")

    # Print summary
    print("")
    print("SUMMARY TABLE (Average inference time in seconds)")
    print("=" * 100)
    for line in report_lines[
        report_lines.index("SUMMARY TABLE (Average inference time in seconds)") + 1 :
    ]:
        print(line)

    # Save to file
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nResults saved to: {output_path.absolute()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Style-Bert-VITS2 dtype comparison benchmark"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use for inference (default: cuda)",
    )
    parser.add_argument(
        "--model",
        default="amitaro",
        help="Model name to use (default: amitaro)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per test case (default: 3)",
    )
    parser.add_argument(
        "--output",
        default="tests/benchmark_dtype_results.txt",
        help="Output file path (default: tests/benchmark_dtype_results.txt)",
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = "cpu"

    try:
        run_benchmark(
            device=args.device,
            model_name=args.model,
            num_runs=args.runs,
            output_file=args.output,
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    except Exception as e:
        logger.exception(f"Error during benchmark: {e}")


if __name__ == "__main__":
    main()
