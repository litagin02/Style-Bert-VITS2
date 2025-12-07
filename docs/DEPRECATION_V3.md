# Style-Bert-VITS2 v3.0.0 - Japanese-Only Architecture

## Overview

Starting with v3.0.0, Style-Bert-VITS2 supports **Japanese (JP) only**. Chinese (ZH) and English (EN) language support has been removed.

This document explains the changes and provides migration guidance.

## Why This Change?

1. **Simplified Architecture**: The JP-Extra model architecture provides better quality for Japanese text-to-speech compared to the multilingual model.
2. **Reduced Complexity**: Removing multilingual support simplifies the codebase, making it easier to maintain and extend.
3. **Focus on Quality**: By concentrating on Japanese, we can optimize performance and quality for the primary use case.
4. **Resource Efficiency**: Smaller model size due to reduced vocabulary and embedding layers.

## Breaking Changes

### Removed Features

1. **Chinese (ZH) Language Support**
   - `nlp/chinese/` directory removed
   - ZH BERT models no longer loaded
   - `freeze_ZH_bert` training parameter has no effect

2. **English (EN) Language Support**
   - `nlp/english/` directory removed
   - EN BERT models no longer loaded
   - `freeze_EN_bert` training parameter has no effect

3. **Multilingual Model Architecture**
   - The old multilingual `models/models.py` (containing multilingual `SynthesizerTrn`) has been removed
   - `models/models.py` now contains only the JP-Extra architecture (renamed from `models_jp_extra.py`)

4. **Training Script**
   - `scripts/train.py` now contains only the JP-Extra training (renamed from `train_jp_extra.py`)
   - The old multilingual training script has been removed

### API Changes

- The `language` parameter in API calls now only accepts `"JP"`. Other values will raise an error.
- The `Languages` enum now only contains `JP`.

## Migration Guide

### For Users with JP-Extra Models

If you were already using JP-Extra models (which is the recommended model type), your models will continue to work. However, we recommend converting them to the v3.0 format for optimal performance:

```bash
# Convert a single model
python scripts/convert_model_v3.py --input model.safetensors --output model_v3.safetensors

# Convert all models in a directory
python scripts/convert_model_v3.py --input model_assets/
```

The conversion script:
1. Reduces the embedding vocabulary from ~112 symbols to ~52 symbols (JP-only)
2. Reduces tone embeddings from 12 (ZH+JP+EN) to 2 (JP-only)
3. Reduces language embeddings from 3 to 1

### For Users with Multilingual Models

If you have multilingual (non-JP-Extra) models:
1. These models are no longer supported in v3.0.0
2. Consider using v2.x for multilingual support
3. For Japanese-only use, retrain with JP-Extra architecture

### Training Migration

Training uses `train.py` (which now contains only JP-Extra training):

```bash
python train.py -c config.json -m model_dir
```

### API Migration

```python
# Old (multilingual)
from style_bert_vits2.constants import Languages
result = model.infer(text, language=Languages.EN)  # No longer works

# New (JP-only)
from style_bert_vits2.constants import Languages
result = model.infer(text, language=Languages.JP)  # This is the only option
result = model.infer(text)  # language defaults to JP
```

## Symbol Vocabulary Changes

| Version | Total Symbols | Languages | Tones |
|---------|---------------|-----------|-------|
| v2.x    | ~112          | ZH, JP, EN| 12    |
| v3.0    | 52            | JP only   | 2     |

The v3.0 symbol set is a subset of v2.x, containing only Japanese phonemes, punctuation, and special tokens.

## Backward Compatibility

### Loading Pre-3.x Models

Pre-3.x JP-Extra models can still be loaded in v3.0. The system automatically detects the vocabulary size from the model file:

- If `enc_p.emb.weight` has 112 symbols: Uses legacy SYMBOLS_V2
- If `enc_p.emb.weight` has 52 symbols: Uses new SYMBOLS_V3

This allows seamless loading of existing models without conversion.

### Converting Models (Recommended)

While pre-3.x models work, we recommend converting them to v3.0 format:

1. Smaller file size (fewer parameters in embedding layers)
2. Consistent with new training
3. Avoid potential edge cases

## Files Changed

### Removed Files

- `style_bert_vits2/nlp/chinese/` (entire directory)
- `style_bert_vits2/nlp/english/` (entire directory)
- Old multilingual `style_bert_vits2/models/models.py`
- Old multilingual `scripts/train.py`

### Renamed Files

- `style_bert_vits2/models/models_jp_extra.py` → `style_bert_vits2/models/models.py`
- `scripts/train_jp_extra.py` → `scripts/train.py`

### Modified Files

- `style_bert_vits2/constants.py` - Languages enum now only contains JP
- `style_bert_vits2/nlp/__init__.py` - JP-only processing
- `style_bert_vits2/nlp/bert_models.py` - Simplified BERT loading
- `style_bert_vits2/nlp/symbols.py` - Added SYMBOLS_V2/V3 versioning
- `style_bert_vits2/models/infer.py` - Vocab detection, JP-only
- `style_bert_vits2/models/infer_onnx.py` - JP-only
- `style_bert_vits2/tts_model.py` - JP-only
- `scripts/convert_onnx.py` - JP-Extra only

### New Files

- `scripts/convert_model_v3.py` - Model conversion script
- `docs/DEPRECATION_V3.md` - This document

## FAQ

**Q: Can I still use Chinese or English text?**
A: No, v3.0 only supports Japanese. For multilingual support, use v2.x.

**Q: Will my existing JP-Extra model work?**
A: Yes, pre-3.x JP-Extra models are automatically detected and work correctly.

**Q: Should I convert my models to v3.0 format?**
A: It's recommended but not required. Conversion reduces file size and ensures compatibility.

**Q: What about the freeze_EN_bert and freeze_ZH_bert training options?**
A: These options are ignored in JP-Extra training as there are no EN/ZH BERT projections.

**Q: How do I train a new model?**
A: Use `train.py` which now contains only JP-Extra training (the old multilingual training script has been removed).

## Support

For questions or issues related to the v3.0 migration:
- Open an issue on GitHub
- Check the documentation and FAQ

For continued multilingual support, consider using the v2.x branch.
