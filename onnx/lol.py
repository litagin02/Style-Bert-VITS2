from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer


tokenizer = Tokenizer.from_file("tokenizer.json")
tokenizer.pre_tokenizer = BertPreTokenizer()
inputs = tokenizer.encode("おはようございます")
print(input)
