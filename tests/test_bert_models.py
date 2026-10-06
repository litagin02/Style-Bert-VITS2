import pytest
import torch
from transformers import DebertaV2Config, DebertaV2ForMaskedLM

from style_bert_vits2.constants import Languages
from style_bert_vits2.nlp import bert_models


@pytest.mark.parametrize("language", [Languages.JP, Languages.EN])
def test_load_model_is_float32(language: Languages, tmp_path):
    """
    float16 で保存された BERT モデルでも、load_model() は float32 で読み込む。

    transformers 5 以降は from_pretrained() の dtype のデフォルトが "auto" になり、
    config.json に記載された dtype (float16) のまま読み込まれて CPU 推論に失敗する。
    実際の BERT モデルのダウンロードを避けるため、小さなダミーモデルで検証する。
    """

    config = DebertaV2Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    DebertaV2ForMaskedLM(config).half().save_pretrained(tmp_path)

    try:
        model = bert_models.load_model(language, str(tmp_path))
        assert next(model.parameters()).dtype == torch.float32
    finally:
        bert_models.unload_model(language)
