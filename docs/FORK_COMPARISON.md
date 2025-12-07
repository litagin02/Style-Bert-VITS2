# Comparison with Style-Bert-VITS2 fork

Reference fork: `reference/Style-Bert-VITS2` (community fork around v2.7.0, `hatchling`, requirements files).
Current repo: `SBV2-gemini` (original project, v3.0.0.dev0, `uv` backend).

## Integration Status

### Completed

1. **Text processing**
   - Ported improved `style_bert_vits2/nlp/japanese/normalizer.py` with unit/date/currency handling
   - Added `katakana_map.py` for katakana normalization
   - Ported `mora_list.py` with pyopenjtalk-plus morae support
   - Ported robust `adjust_word2ph()` fallback in `g2p.py`
   - Ported mora validation rules in `user_dict/word_model.py`

2. **G2P improvements**
   - Added 15+ morae from pyopenjtalk-plus (フュ, デェ, グォ/グェ/グゥ/グィ, クォ/クェ/クゥ/クィ, ヂャ/ヂュ/ヂョ/ヂェ, シィ, グァ, クァ)
   - Better handling of loan words (クォーター, フュージョン, etc.)
   - Robust word2ph adjustment when generated vs given phone lengths differ significantly

3. **Inference optimizations**
   - Ported `infer_stream` for streaming TTS output
   - Implemented `model_dtype` parameter for FP16/BF16 inference (replaces old `use_fp16`)
   - Chunk-based decoder processing with overlap-add for smooth streaming
   - Gradio streaming with chunk accumulation for smooth playback

4. **API endpoints**
   - Added `/voice/stream` streaming endpoint to `server_fastapi.py`
   - Added streaming button to Gradio UI with configurable chunk/overlap size

5. **Benchmarking**
   - `tests/benchmark_dtype.py` - FP32/FP16/BF16 comparison benchmark
   - `tests/streaming_benchmark.py` - Streaming vs normal inference benchmark
   - `tests/test_normalizer.py` - Japanese text normalizer tests

6. **English NLP**
   - Ported `short_form_dict` for abbreviations (89 entries: NASA, FBI, GPT, API, CPU, GPU, etc.)
   - Added `get_shortform_dict()` to `cmudict.py`
   - Updated `g2p.py` to check short_form_dict first for uppercase words

### Not Ported (by design)

- **ONNX inference**: v3.0.0 is PyTorch-only for runtime inference
- **Local BERT assets**: Using remote Hugging Face models instead
- **faster_whisper**: Using transformers Whisper only
- **Dockerfiles**: Not needed for current deployment model
- **bert_models.py quantization**: Fork has `use_fp16`/`use_int8` options, current uses `model_dtype` approach

## High-level deltas

| Aspect | Fork | Current (v3.0.0) |
|--------|------|------------------|
| Packaging | `hatch` + requirements.txt | `uv` only |
| Torch version | <2.9 | 2.9.x |
| BERT assets | Local in `bert/` | Remote HuggingFace |
| Inference | ONNX + PyTorch | PyTorch only |
| Streaming | `infer_stream` | Ported |
| Precision | `use_fp16` bool | `model_dtype` (FP16/BF16/None) |
| Speech-to-text | faster_whisper | transformers Whisper |

## Key implementation differences

### Precision control

**Fork approach:**
```python
# Boolean flag
TTSModelHolder(..., use_fp16=True)
model.infer(..., use_fp16=True)
```

**Current approach:**
```python
# Explicit dtype
TTSModelHolder(..., model_dtype=torch.float16)  # or torch.bfloat16
# Dtype is applied at model load, not per-inference
```

### Streaming

**Fork approach:**
- Returns raw float32 audio chunks
- No chunk accumulation

**Current approach:**
- Returns int16 PCM audio chunks (consistent with `infer()`)
- Gradio: Accumulates chunks to 1+ second for smooth playback
- API: Streams WAV with header + PCM chunks

## Files modified from fork

| File | Changes |
|------|---------|
| `style_bert_vits2/models/infer.py` | Added `infer_stream`, `model_dtype` support, vocab detection |
| `style_bert_vits2/models/models.py` | Added dtype handling in inference (JP-Extra only, renamed from `models_jp_extra.py`) |
| `style_bert_vits2/tts_model.py` | Added `infer_stream`, `model_dtype`, updated `convert_to_16_bit_wav` |
| `style_bert_vits2/nlp/japanese/normalizer.py` | Ported improved normalization |
| `style_bert_vits2/nlp/japanese/mora_list.py` | Added pyopenjtalk-plus morae |
| `style_bert_vits2/nlp/japanese/g2p.py` | Added robust `adjust_word2ph()` fallback |
| `style_bert_vits2/nlp/japanese/user_dict/__init__.py` | Load all `dict_data/**/*.dic` files (aivis_dictionaries) |
| `style_bert_vits2/nlp/japanese/user_dict/word_model.py` | Updated mora validation rules |
| `style_bert_vits2/nlp/japanese/pyopenjtalk_worker/__init__.py` | Accept list of dictionary paths, auto-reconnect |
| `scripts/gradio_tabs/inference.py` | Added streaming UI with chunk config |
| `scripts/server_fastapi.py` | Added `/voice/stream` endpoint |

## Detailed File Comparison

### Japanese NLP (`style_bert_vits2/nlp/japanese/`)

| File | Current vs Fork |
|------|-----------------|
| `mora_list.py` | Fork adds 15+ morae from pyopenjtalk-plus |
| `g2p.py` | Fork has typed `NJDFeature`/`OpenJTalk`, robust fallback |
| `bert_feature.py` | Current has `dtype` parameter (better) |
| `user_dict/__init__.py` | Current loads `dict_data/**/*.dic` (better) |
| `user_dict/word_model.py` | Fork has expanded mora validation |
| `pyopenjtalk_worker/` | Current has auto-reconnect (better) |

### Top-level NLP (`style_bert_vits2/nlp/`)

| File | Current vs Fork |
|------|-----------------|
| `__init__.py` | Current has `dtype` parameter (better) |
| `bert_models.py` | Fork has `device_map`, `use_fp16`, `use_int8` quantization |

### English NLP (`style_bert_vits2/nlp/english/`)

| File | Current vs Fork |
|------|-----------------|
| `g2p.py` | Ported - checks short_form_dict for uppercase words |
| `cmudict.py` | Ported - added `get_shortform_dict()` |
| `short_form_dict.rep` | Ported - 89 abbreviation entries |
