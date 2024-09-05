from transformers.convert_slow_tokenizer import BertConverter
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages


bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

tokenizer = bert_models.load_tokenizer(Languages.JP)
converter = BertConverter(tokenizer)
model = converter.converted()


model.save("tokenizer.json")
