import re
import sys
import unicodedata
from datetime import datetime

from e2k import C2K, NGram
from jaconv import jaconv
from num2words import num2words

from style_bert_vits2.nlp.japanese.katakana_map import KATAKANA_MAP
from style_bert_vits2.nlp.symbols import PUNCTUATIONS


# C2K / NGram の初期化
# NGram は英単語として読ませるか、アルファベット読みするべきかを判定するモデル
__characters_to_katakana = C2K()
__should_transliterated_word_by_ngram = NGram()

# 記号類の正規化マップ
__SYMBOL_REPLACE_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "．": ".",
    "…": "...",
    "···": "...",
    "・・・": "...",
    "/": "/",  # スラッシュは pyopenjtalk での形態素解析処理で重要なので正規化後も残す
    "／": "/",  # スラッシュは pyopenjtalk での形態素解析処理で重要なので正規化後も残す
    "\\": ".",
    "＼": ".",
    "·": ",",
    "・": ",",
    "、": ",",
    "$": ".",
    ":": ",",
    ";": ",",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "「": "'",
    "」": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    # NFKC 正規化後のハイフン・ダッシュの変種を全て通常半角ハイフン - \u002d に変換
    "\u02d7": "\u002d",  # ˗, Modifier Letter Minus Sign
    "\u2010": "\u002d",  # ‐, Hyphen,
    # "\u2011": "\u002d",  # ‑, Non-Breaking Hyphen, NFKC により \u2010 に変換される
    "\u2012": "\u002d",  # ‒, Figure Dash
    "\u2013": "\u002d",  # –, En Dash
    "\u2014": "\u002d",  # —, Em Dash
    "\u2015": "\u002d",  # ―, Horizontal Bar
    "\u2043": "\u002d",  # ⁃, Hyphen Bullet
    "\u2212": "\u002d",  # −, Minus Sign
    "\u23af": "\u002d",  # ⎯, Horizontal Line Extension
    "\u23e4": "\u002d",  # ⏤, Straightness
    "\u2500": "\u002d",  # ─, Box Drawings Light Horizontal
    "\u2501": "\u002d",  # ━, Box Drawings Heavy Horizontal
    "\u2e3a": "\u002d",  # ⸺, Two-Em Dash
    "\u2e3b": "\u002d",  # ⸻, Three-Em Dash
    # "～": "-",  # これは長音記号「ー」として扱うよう変更
    # "~": "-",  # これも長音記号「ー」として扱うよう変更
}
# 記号類の正規化パターン
__SYMBOL_REPLACE_PATTERN = re.compile(
    "|".join(re.escape(p) for p in __SYMBOL_REPLACE_MAP)
)

# 記号などの読み正規化マップ
# 一度リストアップしたがユースケース上不要と判断した記号はコメントアウトされている
__SYMBOL_YOMI_MAP = {
    # 一般記号
    # "@": "アット",  # カタカナに変換するとユーザー辞書登録が効かなくなり不便なためコメントアウト
    # "＠": "アット",  # カタカナに変換するとユーザー辞書登録が効かなくなり不便なためコメントアウト
    # "&": "アンド",  # カタカナに変換するとユーザー辞書登録が効かなくなり不便なためコメントアウト
    # "＆": "アンド",  # カタカナに変換するとユーザー辞書登録が効かなくなり不便なためコメントアウト
    # "*": "アスタリスク",  # 発音されると不都合な場合があるためコメントアウト
    # "＊": "アスタリスク",  # 発音されると不都合な場合があるためコメントアウト
    "#": "シャープ",
    "＃": "シャープ",
    "#️⃣": "シャープ",
    "†": "ダガー",
    "‡": "ダブルダガー",
    "§": "セクション",
    "¶": "パラグラフ",
    # 算術演算子
    "+": "プラス",
    "＋": "プラス",
    "➕": "プラス",
    "➖": "マイナス",  # 絵文字以外のハイフンは伸ばす棒と区別がつかないので記述していない
    "×": "かける",
    "✖": "かける",
    "⨯": "かける",
    "÷": "わる",
    "➗": "わる",
    # 等号・不等号
    "=": "イコール",
    "＝": "イコール",
    "≠": "ノットイコール",
    "≒": "ニアリーイコール",
    "≈": "ニアリーイコール",
    "≅": "合同",
    "≡": "合同",
    "≢": "合同でない",
    # 比較演算子
    # 山括弧は装飾的に使われることも多いため、別途数式や比較演算子として使われる場合のみ読み上げる
    "≤": "小なりイコール",
    "≦": "小なりイコール",
    "⩽": "小なりイコール",
    "≥": "大なりイコール",
    "≧": "大なりイコール",
    "⩾": "大なりイコール",
    # 単位・数値記号
    "%": "パーセント",
    "％": "パーセント",
    "٪": "パーセント",
    "﹪": "パーセント",
    "‰": "パーミル",
    "‱": "パーミリアド",
    "′": "プライム",
    "″": "ダブルプライム",
    "‴": "トリプルプライム",
    "°": "度",
    "℃": "度",
    "℉": "度",
    "±": "プラスマイナス",
    "∓": "マイナスプラス",
    "№": "ナンバー",
    "℡": "電話番号",
    "〒": "郵便番号",
    "〶": "郵便番号",
    "㏍": "株式会社",
    "℠": "エスエム",
    # "™": "ティーエム",
    "©": "コピーライト",
    # "®": "アールマーク",
    "💲": "$",  # __convert_numbers_to_words() で「100ドル」のように読み上げできるように
    # 音楽記号
    "♯": "シャープ",
    "♭": "フラット",
    "♮": "ナチュラル",
    # "♩": "音符",
    # "♪": "音符",
    # "♫": "音符",
    # "♬": "音符",
    # 数学記号
    "∧": "かつ",
    "∨": "または",
    "¬": "ノット",
    "⊕": "排他的論理和",
    "⊗": "テンソル積",
    "√": "ルート",
    "∛": "立方根",
    "∜": "四乗根",
    "∞": "無限大",
    "♾️": "無限大",
    "π": "パイ",
    "∑": "シグマ",
    "∏": "パイ積分",
    "∫": "インテグラル",
    "∬": "二重積分",
    "∭": "三重積分",
    "∮": "周回積分",
    "∯": "面積分",
    "∰": "体積分",
    "∂": "パーシャル",
    "∇": "ナブラ",
    "∝": "比例",
    # 集合記号
    "∈": "属する",
    "∉": "属さない",
    "∋": "含む",
    "∌": "含まない",
    "∪": "和集合",
    "∩": "共通部分",
    "⊂": "部分集合",
    "⊃": "上位集合",
    "⊄": "部分集合でない",
    "⊅": "上位集合でない",
    "⊆": "部分集合または等しい",
    "⊇": "上位集合または等しい",
    "∅": "空集合",
    "∖": "差集合",
    "∆": "対称差",
    # 幾何記号
    "∥": "平行",
    "⊥": "垂直",
    "∠": "角",
    "∟": "直角",
    "∡": "測定角",
    "∢": "球面角",
    # 囲み付き・丸付き文字 (Unicode 正規化で問題なく変換される「㈦」「㊥」などを除く)
    # ref: https://ja.wikipedia.org/wiki/%E5%9B%B2%E3%81%BFCJK%E6%96%87%E5%AD%97%E3%83%BB%E6%9C%88
    "㈱": "株式会社",
    "㈲": "有限会社",
    "㈳": "社団法人",
    "㈴": "合名会社",
    "㈵": "特殊法人",
    "㈶": "財団法人",
    "㈷": "祝日",
    "㈸": "労働組合",
    "㈹": "代表電話",
    "㈺": "呼出し電話",
    "㈻": "学校法人",
    "㈼": "監査法人",
    "㈽": "企業組合",
    "㈾": "合資会社",
    "㈿": "協同組合",
    "㉀": "祭日",
    "㉁": "休日",
    "㉅": "幼稚園",
    "㊑": "株式会社",
    "㊒": "有限会社",
    "㊓": "社団法人",
    "㊔": "合名会社",
    "㊕": "特殊法人",
    "㊖": "財団法人",
    "㊗": "祝日",
    "㊘": "労働組合",
    "㊙": "マル秘",
    "㊝": "マル優",
    "㊡": "休日",
    "㊢": "写し",
    "㊩": "医療法人",
    "㊪": "宗教法人",
    "㊫": "学校法人",
    "㊬": "監査法人",
    "㊭": "企業組合",
    "㊮": "合資会社",
    "㊯": "協同組合",
}
# 記号類の読み正規化パターン
__SYMBOL_YOMI_PATTERN = re.compile("|".join(re.escape(p) for p in __SYMBOL_YOMI_MAP))

# 単位の正規化マップ
# 単位は OpenJTalk 側で変換してくれるものもあるため、単位が1文字で読み間違いが発生しやすい L, m, g, B と、
# OpenJTalk では変換できない単位、正規化処理で変換しておいた方が実装上都合が良い単位のみ変換する
__UNIT_MAP = {
    "kL": "キロリットル",
    "dL": "デシリットル",
    "mL": "ミリリットル",
    "L": "リットル",
    "km": "キロメートル",
    "km2": "平方キロメートル",
    "km3": "立方キロメートル",
    "m2": "平方メートル",
    "m3": "立方メートル",
    "cm": "センチメートル",
    "cm2": "平方センチメートル",
    "cm3": "立方センチメートル",
    "mm": "ミリメートル",
    "mm2": "平方ミリメートル",
    "mm3": "立方ミリメートル",
    "m": "メートル",
    "kg": "キログラム",
    "mg": "ミリグラム",
    "g": "グラム",
    "EB": "エクサバイト",
    "EiB": "エクスビバイト",
    "PB": "ペタバイト",
    "PiB": "ペビバイト",
    "TB": "テラバイト",
    "TiB": "テビバイト",
    "GB": "ギガバイト",
    "GiB": "ギビバイト",
    "MB": "メガバイト",
    "MiB": "メビバイト",
    "KB": "キロバイト",
    "kB": "キロバイト",
    "KiB": "キビバイト",
    "B": "バイト",
    "mA": "ミリアンペア",
    "kA": "キロアンペア",
    "A": "アンペア",
    "t": "トン",
    "d": "日",
    "h": "時間",
    "s": "秒",
    "ms": "ミリ秒",
    "μs": "マイクロ秒",
    "ns": "ナノ秒",
    "THz": "テラヘルツ",
    "GHz": "ギガヘルツ",
    "MHz": "メガヘルツ",
    "kHz": "キロヘルツ",
    "KHz": "キロヘルツ",
    "Hz": "ヘルツ",
    "Thz": "テラヘルツ",
    "Ghz": "ギガヘルツ",
    "Mhz": "メガヘルツ",
    "khz": "キロヘルツ",
    "Khz": "キロヘルツ",
    "hz": "ヘルツ",
    "hPa": "ヘクトパスカル",
    "hpa": "ヘクトパスカル",
    "HPa": "ヘクトパスカル",
    "Ebps": "エクサビーピーエス",
    "Pbps": "ペタビーピーエス",
    "Tbps": "テラビーピーエス",
    "Gbps": "ギガビーピーエス",
    "Mbps": "メガビーピーエス",
    "Kbps": "キロビーピーエス",
    "kbps": "キロビーピーエス",
    "bps": "ビーピーエス",
    "Ebit": "エクサビット",
    "Pbit": "ペタビット",
    "Tbit": "テラビット",
    "Gbit": "ギガビット",
    "Mbit": "メガビット",
    "Kbit": "キロビット",
    "kbit": "キロビット",
    "bit": "ビット",
    "Eb": "エクサビット",
    "Pb": "ペタビット",
    "Tb": "テラビット",
    "Gb": "ギガビット",
    "Mb": "メガビット",
    "Kb": "キロビット",
    "kb": "キロビット",
    "b": "ビット",
}
# 単位の正規化パターン
__UNIT_PATTERN = re.compile(
    r"(?P<number>[0-9.]*[0-9](?:[eE][-+]?[0-9]+)?)\s*"
    r"(?P<unit>(?:(k|d|m)?L|(?:k|c|m)m[23]?|m[23]?|m(?![a-zA-Z])|"
    r"(?:k|m)?g|(?:k|K|M|G|T|P|E)(?:i)?B|B|t|d|h|s|ms|μs|ns|"
    r"(?:k|m)?A|(?:k|K|M|G|T)?[Hh]z|[Hh][Pp]a|(?:k|K|M|G|T|P|E)?(?:bps|bit|b)))"
    r"(?P<suffix>/[hs])?"
    r"(?=($|(?=/([^A-Za-z]|$))|[^/A-Za-z]))"
)

# 正規化後に残す文字種を表すパターン
__PUNCTUATION_CLEANUP_PATTERN = re.compile(
    # ↓ ひらがな、カタカナ、漢字
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"
    # ↓ 半角数字
    + r"\u0030-\u0039"
    # ↓ 全角数字
    + r"\uFF10-\uFF19"
    # ↓ 半角アルファベット（大文字と小文字）
    + r"\u0041-\u005A\u0061-\u007A"
    # ↓ 全角アルファベット（大文字と小文字）
    + r"\uFF21-\uFF3A\uFF41-\uFF5A"
    # ↓ ギリシャ文字
    + r"\u0370-\u03FF\u1F00-\u1FFF"
    # ↓ "!", "?", "…", ",", ".", "'", "-", 但し`…`はすでに`...`に変換されている
    # スラッシュは pyopenjtalk での形態素解析処理で重要なので、例外的に正規化後も残す (g2p 処理内で "." に変換される)
    # pyopenjtalk は「漢字の直後に2つ以上の連続する半角ハイフンがある場合」にその漢字の読みが取得できなくなる謎のバグがあるため、
    # 正規化処理でダッシュが変換されるなどして2つ以上の連続する半角ハイフンが生まれた場合、Long EM Dash に変換してから g2p 処理に渡す
    + "".join(re.escape(p) for p in (PUNCTUATIONS + ["/", "—"]))
    + r"]+"
)

# 数字・通貨記号の正規化パターン
__CURRENCY_MAP = {
    "$": "ドル",
    "¥": "円",
    "€": "ユーロ",
    "£": "ポンド",
    "₩": "ウォン",
    "₹": "ルピー",  # インド・ルピー
    "₽": "ルーブル",
    "₺": "リラ",  # トルコ・リラ
    "฿": "バーツ",
    "₱": "ペソ",  # フィリピン・ペソ
    "₴": "フリヴニャ",
    "₫": "ドン",
    "₪": "シェケル",  # イスラエル・新シェケル
    "₦": "ナイラ",
    "₡": "コロン",  # コスタリカ・コロン
    "₿": "ビットコイン",
    "﷼": "リヤル",  # サウジアラビア・リヤル
    "₠": "ECU",  # European Currency Unit (廃止)
    "₢": "クルザード",  # ブラジル・クルザード (廃止)
    "₣": "フランスフラン",  # フランス・フラン (廃止)
    "₤": "リラ",  # イタリア・リラ (廃止)
    "₥": "ミル",  # アメリカ・ミル (廃止)
    "₧": "ペセタ",  # スペイン・ペセタ (廃止)
    "₨": "ルピー",  # パキスタン・ルピー
    "₭": "キープ",  # ラオス・キープ
    "₮": "トゥグルグ",  # モンゴル・トゥグルグ
    "₯": "ドラクマ",  # ギリシャ・ドラクマ (廃止)
    "₰": "ドイツペニヒ",  # ドイツ・ペニヒ (廃止)
    "₲": "グアラニー",  # パラグアイ・グアラニー
    "₳": "アウストラール",  # アルゼンチン・アウストラール (廃止)
    "₵": "セディ",  # ガーナ・セディ
    "₶": "リヴルトゥールヌワ",  # フランス・リヴルトゥールヌワ (廃止)
    "₷": "スペルリング",  # マルタ・スペルリング (廃止)
    "₸": "テンゲ",  # カザフスタン・テンゲ
    "₻": "マナト",  # トルクメニスタン・マナト
    "₼": "アゼルバイジャンマナト",
    "₾": "ラリ",  # ジョージア・ラリ
}
__CURRENCY_PATTERN = re.compile(
    r"([$¥€£₩₹₽₺฿₱₴₫₪₦₡₿﷼₠₢₣₤₥₧₨₭₮₯₰₲₳₵₶₷₸₻₼₾])([0-9.]*[0-9])|([0-9.]*[0-9])([$¥€£₩₹₽₺฿₱₴₫₪₦₡₿﷼₠₢₣₤₥₧₨₭₮₯₰₲₳₵₶₷₸₻₼₾])"
)
__NUMBER_PATTERN = re.compile(r"[0-9]+(\.[0-9]+)?")
__NUMBER_WITH_SEPARATOR_PATTERN = re.compile("[0-9]{1,3}(,[0-9]{3})+")

# __replace_symbols() で使う正規表現パターン
__DATE_ZERO_PADDING_PATTERN = re.compile(r"(?<!\d)0(\d)(?=月|日|時|分|秒)")
__TIME_PATTERN = re.compile(r"(\d+)時(\d+)分(?:(\d+)秒)?")
__ASPECT_PATTERN = re.compile(r"(\d+)[:：](\d+)(?:[:：](\d+))?")
__WEEKDAY_PATTERN = re.compile(
    r"("  # 日付部分をキャプチャ開始
    r"(?:\d{4}年\s*)?"  # 4桁の年 + 年（省略可）
    r"(?:\d{1,2}月\s*)?"  # 1-2桁の月 + 月（省略可）
    r"\d{1,2}日"  # 1-2桁の日 + 日（必須）
    r")"  # 日付部分をキャプチャ終了
    r"\s*[（(]([月火水木金土日])[)）]"  # 全角/半角括弧で囲まれた曜日漢字
    r"|"  # または
    r"("  # 日付部分をキャプチャ開始
    r"(?:\d{4}[-/]\s*)?"  # 4桁の年 + 区切り（省略可）
    r"(?:\d{1,2}[-/]\s*)?"  # 1-2桁の月 + 区切り（省略可）
    r"\d{1,2}"  # 1-2桁の日（必須）
    r")"  # 日付部分をキャプチャ終了
    r"\s*[（(]([月火水木金土日])[)）]"  # 全角/半角括弧で囲まれた曜日漢字
)
__URL_PATTERN = re.compile(
    r"https?://[-a-zA-Z0-9.]+(?:/[-a-zA-Z0-9._~:/?#\[\]@!$&\'()*+,;=]*)?"
)
__EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
# 英数・かな・カナ・漢字のワード文字を判定するパターン
__WORD_CHAR_PATTERN = re.compile(r"[A-Za-z0-9\u3040-\u30FF\u4E00-\u9FFF]")
# 候補記号と空白のみで構成される3文字以上の塊を粗抽出
__BLOCK_PATTERN = re.compile(r"(?:(?:[#$%&*+\-=_:/\\|;<>^])|\s){3,}")
__NUMBER_RANGE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?(?:\s*[a-zA-Z]+)?)\s*[〜~～ー]\s*(\d+(?:\.\d+)?(?:\s*[a-zA-Z]+)?)"
)
__NUMBER_MATH_PATTERN = re.compile(
    r"(\d+)\s*([+＋➕\-−－ー➖×✖⨯÷➗*＊])\s*(\d+)\s*=\s*(\d+)"
)
__NUMBER_COMPARISON_PATTERN = re.compile(r"(\d+)\s*([<＜>＞])\s*(\d+)")
__YEAR_MONTH_PATTERN = re.compile(r"(?<!\d)(18|19|20|21|22)(\d{2})/([0-1]?\d)(?!\d)")
__FRACTION_PATTERN = re.compile(r"(\d+)[/／](\d+)")
__ZERO_HOUR_PATTERN = re.compile(r"(?<![0-9])(午前|午後)?0時(?![0-9分]|間)")
__WAREKI_PATTERN = re.compile(r"([RHS])(\d{1,2})\.(\d{1,2})\.(\d{1,2})")
__DATE_EXPAND_PATTERN = re.compile(r"\d{2}[-/\.]\d{1,2}[-/\.]\d{1,2}")
__DATE_PATTERN = re.compile(
    r"(?<!\d)(?:\d{4}[-/\.][0-9]{1,2}[-/\.][0-9]{1,2}|\d{2}[-/\.][0-9]{1,2}[-/\.][0-9]{1,2}|[0-9]{1,2}/[0-9]{1,2}|\d{4}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))(?!\d)"
)
__EXPONENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)[eE]([-+]?\d+)")

# __convert_english_to_katakana() で使う正規表現パターン
__ENGLISH_WORD_PATTERN = re.compile(r"[a-zA-Z0-9]")
__ENGLISH_WORD_WITH_NUMBER_PATTERN = re.compile(
    r"([a-zA-Z]+)[\s-]?([1-9]|1[01])(?!\d|\.\d)"  # 12 以降は英語読みしない
)
__ALPHABET_PATTERN = re.compile(r"[a-zA-Z]")


def normalize_text(text: str) -> str:
    """
    日本語のテキストを正規化する。
    結果は、ちょうど次の文字のみからなる：
    - ひらがな
    - カタカナ（全角長音記号「ー」が入る！）
    - 漢字
    - 半角数字
    - 半角アルファベット（大文字と小文字）
    - ギリシャ文字
    - `.` （句点`。`や`…`の一部や改行等）
    - `,` （読点`、`や`:`等）
    - `?` （疑問符`？`）
    - `!` （感嘆符`！`）
    - `'` （`「`や`」`等）
    - `-` （`―`（ダッシュ、長音記号ではない）や`-`等）
    - `/` （スラッシュは pyopenjtalk での形態素解析処理で重要なので、例外的に正規化後も残し、g2p 処理内で "." に変換される）
    - `—` （pyopenjtalk のバグ回避のために例外的に正規化後も残し、g2p 処理内で "-" に変換される）

    注意点:
    - 三点リーダー`…`は`...`に変換される（`なるほど…。` → `なるほど....`）
    - 読点や疑問符等の位置・個数等は保持される（`??あ、、！！！` → `??あ,,!!!`）

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    # 最初にカタカナを除く英数字記号 (ASCII 文字) を半角に変換する
    # どのみち Unicode 正規化で行われる処理ではあるが、__replace_symbols() は Unicode 正規化前に実行しなければ正常に動作しない
    # 一方で __replace_symbols() は半角英数字の入力を前提に実装されており、全角英数記号が入ると変換処理（正規表現マッチ）が意図通り実行されない可能性がある
    # これを回避するため、__replace_symbols() の実行前に半角英数記号を半角に変換している
    text = jaconv.z2h(
        text, kana=False, digit=True, ascii=True, ignore="\u3000"
    )  # 全角スペースは変換しない

    # Unicode 正規化前に記号を変換
    # 正規化前でないと ℃ などが unicodedata.normalize() で分割されてしまう
    res = __replace_symbols(text)

    # 自然な日本語テキスト読み上げのために、全角スペースは句点に変換
    # 半角スペースが入る箇所で止めて読むかはケースバイケースなため、変換は行わない
    # Unicode 正規化でスペースが全て半角に変換される前に実行する必要がある
    res = res.replace("\u3000", "。")

    # ゼロ幅スペースを削除
    res = res.replace("\u200b", "")

    res = unicodedata.normalize("NFKC", res)  # ここで Unicode 正規化が行われる

    res = __convert_english_to_katakana(res)  # 英単語をカタカナに変換

    res = __convert_numbers_to_words(res)  # 「100円」→「百円」等
    # 「～」と「〜」と「~」も長音記号として扱う
    res = res.replace("~", "ー")
    res = res.replace("～", "ー")
    res = res.replace("〜", "ー")

    res = replace_punctuation(res)  # 句読点等正規化、読めない文字を削除

    # 結合文字の濁点・半濁点を削除
    # 通常の「ば」等はそのままのこされる、「あ゛」は上で「あ゙」になりここで「あ」になる
    res = res.replace("\u3099", "")  # 結合文字の濁点を削除、る゙ → る
    res = res.replace("\u309a", "")  # 結合文字の半濁点を削除、な゚ → な

    # pyopenjtalk は「漢字の直後に2つ以上の連続する半角ハイフンがある場合」にその漢字の読みが取得できなくなる謎のバグがあるため、
    # 正規化処理でダッシュが変換されるなどして2つ以上の連続する半角ハイフンが生まれた場合、Long EM Dash に変換してから g2p 処理に渡す
    res = re.sub(
        r"([\u4e00-\u9FFF])(-{2,})", lambda m: m.group(1) + "—" * len(m.group(2)), res
    )

    return res


def __replace_symbols(text: str) -> str:
    """
    記号類の読みを適切に変換する。
    この関数は正規化処理の最初に実行する必要がある（さもなければ英数字のカタカナ変換処理の影響を受けてしまう）。
    処理順序によって結果が変わるので無闇に並び替えてはいけない。

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    # 月・日・時・分・秒のゼロ埋めを除去
    text = __DATE_ZERO_PADDING_PATTERN.sub(r"\1", text)

    # 括弧内の曜日表記を変換（日付の後にある場合のみ）
    text = __WEEKDAY_PATTERN.sub(
        lambda m: f"{m.group(1) or m.group(3)}{m.group(2) or m.group(4)}曜日", text
    )

    def convert_url_symbols(match: re.Match[str]) -> str:
        url = match.group(0)
        # 記号を日本語に変換
        # コンマの位置は実際に読み上げた際にちょうど良いテンポ感になるように意図的につけたりつけなかったりしている
        url = url.replace("https://", "エイチティーティーピーエス,")
        url = url.replace("http://", "エイチティーティーピー,")
        url = url.replace(".com", "ドットコム,")
        url = url.replace(".net", "ドットネット,")
        url = url.replace(".org", "ドットオーグ,")
        url = url.replace(".info", "ドットインフォ,")
        url = url.replace(".co.jp", "ドットシーオードットジェイピー,")
        url = url.replace(".jp", "ドットジェイピー,")
        url = url.replace(".", "ドット,")
        url = url.replace("/", ",スラッシュ,")
        url = url.replace("?", ",クエスチョン,")
        url = url.replace("&", ",アンド,")
        url = url.replace("=", "イコール")
        url = url.replace("_", "アンダーバー")
        url = url.replace("-", "ハイフン")
        url = url.replace("#", "シャープ")
        url = url.replace("@", ",アットマーク,")
        url = url.replace(":", "コロン")
        url = url.replace("~", "チルダ")
        url = url.replace("+", "プラス")
        return url.rstrip(",").replace(",,", ",")

    # URL パターンの処理
    text = __URL_PATTERN.sub(convert_url_symbols, text)

    def convert_email_symbols(match: re.Match[str]) -> str:
        email = match.group(0)
        # 記号を日本語に変換
        # コンマの位置は実際に読み上げた際にちょうど良いテンポ感になるように意図的につけたりつけなかったりしている
        email = email.replace("@", ",アットマーク,")
        email = email.replace(".com", "ドットコム")
        email = email.replace(".net", "ドットネット")
        email = email.replace(".org", "ドットオーグ")
        email = email.replace(".info", "ドットインフォ")
        email = email.replace(".co.jp", "ドットシーオードットジェイピー")
        email = email.replace(".jp", "ドットジェイピー")
        email = email.replace(".", "ドット")
        email = email.replace("-", "ハイフン")
        email = email.replace("_", "アンダーバー")
        email = email.replace("+", "プラス")
        return email.rstrip(",").replace(",,", ",")

    # メールアドレスパターンの処理
    text = __EMAIL_PATTERN.sub(convert_email_symbols, text)

    # プレーンテキストの区切りとして使われる連続記号の塊を検出し、句点一つに畳み込む
    # - 強ターゲット: {'#', '-', '_', '=', ':', '*'} は、3個以上でヒット
    # - 弱ターゲット: {'$', '%', '&', '+', '/', '\\', '|', ';', '<', '>', '^'} は、6個以上でヒット
    # - 直前直後が文字（英数・かな・カナ・漢字）の場合は、閾値を引き上げ（5個以上）
    # - 空白は塊の一部として許容する（"___   ___" など）
    def collapse_divider_blocks(src: str) -> str:
        # 許容する候補記号
        divider_strong = set("#-_=:*")
        divider_weak = set("$%&+/\\|;<>^")
        divider_all = divider_strong | divider_weak

        def repl(m: re.Match[str]) -> str:
            block = m.group(0)
            # 非空白の候補記号だけを数える
            nonspace_chars = [c for c in block if not c.isspace()]
            if not nonspace_chars:
                return block
            if not all(c in divider_all for c in nonspace_chars):
                return block

            # 文脈による閾値の調整（両隣がワード文字であれば厳しめ）
            start = m.start()
            end = m.end()
            left_char = src[start - 1] if start > 0 else ""
            right_char = src[end] if end < len(src) else ""
            left_is_word = bool(__WORD_CHAR_PATTERN.match(left_char))
            right_is_word = bool(__WORD_CHAR_PATTERN.match(right_char))
            # 両側がワード: 5、それ以外: 3
            base_threshold = 5 if (left_is_word and right_is_word) else 3

            strong_count = sum(1 for c in nonspace_chars if c in divider_strong)
            total_count = len(nonspace_chars)

            # 強ターゲットが閾値以上、または弱ターゲットのみだが十分な長さ
            if strong_count >= base_threshold:
                return "."
            # 弱ターゲットのみの場合の閾値（より厳しく）
            # 基本閾値に2を加算、ただし最低でも6文字必要
            weak_only_threshold = max(base_threshold + 2, 6)
            if strong_count == 0 and total_count >= weak_only_threshold:
                return "."

            return block

        return __BLOCK_PATTERN.sub(repl, src)

    # 区切り用途の連続記号ブロックを句点に畳み込む
    ## URL・メールアドレスパターンを置換した後に適用する
    ## ここまでで URL・メールの記号は文字列化されるため、"://" のような並びは残っておらず、誤検出を回避しやすい
    ## （この段階で畳み込むことで、後段の記号読み変換に到達する前に区切り線を排除できる）
    text = collapse_divider_blocks(text)

    # 数字の範囲を処理
    def convert_range(match: re.Match[str]) -> str:
        start = match.group(1)
        end = match.group(2)
        # 単位を含む場合は単位も含めて変換
        # __UNIT_MAP の単位に対応
        for unit_abbr, unit_full in __UNIT_MAP.items():
            if unit_abbr in start or unit_full in start:
                # 省略形から完全な形に変換
                converted_start = start.replace(unit_abbr, unit_full)
                converted_end = end.replace(unit_abbr, unit_full)
                return f"{converted_start}から{converted_end}"
        return f"{start}から{end}"

    text = __NUMBER_RANGE_PATTERN.sub(convert_range, text)

    def get_symbol_yomi(symbol: str) -> str:
        # 読み間違いを防ぐため、数式の間に挟まれた場合にのみ下記の通り読み上げる
        if symbol in ("-", "−", "－", "ー"):
            return "マイナス"
        if symbol in ("*", "＊"):
            return "かける"
        return __SYMBOL_YOMI_MAP.get(symbol, symbol)

    def get_comparison_yomi(symbol: str) -> str:
        # 比較演算子の読み方を定義
        if symbol in ("<", "＜"):
            return "小なり"
        if symbol in (">", "＞"):
            return "大なり"
        return symbol

    # 数式を処理
    text = __NUMBER_MATH_PATTERN.sub(
        lambda m: f"{m.group(1)}{get_symbol_yomi(m.group(2))}{m.group(3)}イコール{m.group(4)}",
        text,
    )
    # 比較演算子を処理
    text = __NUMBER_COMPARISON_PATTERN.sub(
        lambda m: f"{m.group(1)}{get_comparison_yomi(m.group(2))}{m.group(3)}", text
    )

    # 和暦の省略表記を変換
    def convert_wareki(match: re.Match[str]) -> str:
        era = match.group(1)  # R/H/S
        year = int(match.group(2))  # 年
        month = int(match.group(3))  # 月
        day = int(match.group(4))  # 日
        # 年の範囲チェック（1-99）
        if not 1 <= year <= 99:
            return match.group(0)
        # 月の範囲チェック（1-12）
        if not 1 <= month <= 12:
            return match.group(0)
        # 日の範囲チェック（1-31）
        if not 1 <= day <= 31:
            return match.group(0)
        # 和暦の変換
        era_map = {
            "R": "令和",
            "H": "平成",
            "S": "昭和",
        }
        if era in era_map:
            return f"{era_map[era]}{year}年{month}月{day}日"
        return match.group(0)

    # 和暦の省略表記のパターン
    # R6.1.1, H31.4.30, S64.1.7 などにマッチ
    text = __WAREKI_PATTERN.sub(convert_wareki, text)

    def date_to_words(match: re.Match[str]) -> str:
        date_str = match.group(0)
        try:
            # 連続した数字形式（YYYYMMDD）の場合
            if len(date_str) == 8 and date_str.isdigit():
                try:
                    date = datetime.strptime(date_str, "%Y%m%d")
                    return f"{date.year}年{date.month}月{date.day}日"
                except ValueError:
                    pass

            # 2桁の年を4桁に拡張する処理 (Y/m/d or Y-m-d or Y.m.d の時のみ)
            if __DATE_EXPAND_PATTERN.match(date_str):
                # スラッシュまたはハイフンまたはドットで分割して年部分を取得
                year_str = (
                    date_str.split("/")[0]
                    if "/" in date_str
                    else (
                        date_str.split("-")[0]
                        if "-" in date_str
                        else date_str.split(".")[0]
                    )
                )
                if len(year_str) == 2:
                    # 50 以降は 1900 年代、49 以前は 2000 年代として扱う
                    # 98/04/11 → 1998/04/11 / 36-01-01 → 2036-01-01
                    year_prefix = "19" if int(year_str) >= 50 else "20"
                    date_str = year_prefix + date_str

            # Y/m/d, Y-m-d, Y.m.d, m/d のパターンを試す
            for fmt in ["%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d", "%m/%d"]:
                try:
                    date = datetime.strptime(date_str, fmt)
                    if fmt == "%m/%d":
                        return f"{date.month}月{date.day}日"
                    return f"{date.year}年{date.month}月{date.day}日"
                except ValueError:
                    continue
            # どのパターンにも一致しない場合は元の文字列を返す
            return date_str
        except Exception:
            # エラーが発生した場合は元の文字列を返す
            return date_str

    # 日付パターンの変換
    text = __DATE_PATTERN.sub(date_to_words, text)

    # 年/月形式の処理（1800-2200年の範囲で、かつ月が1-12の場合のみ）
    def convert_year_month(match: re.Match[str]) -> str:
        year = int(f"{match.group(1)}{match.group(2)}")
        month = int(match.group(3))
        # 月が1-12の範囲外の場合は分数として処理するため、元の文字列を返す
        if not 1 <= month <= 12:
            return match.group(0)
        return f"{year}年{month}月"

    # 年/月パターンの変換
    text = __YEAR_MONTH_PATTERN.sub(convert_year_month, text)

    # 分数の処理
    def convert_fraction(match: re.Match[str]) -> str:
        try:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            return f"{num2words(denominator, lang='ja')}ぶんの{num2words(numerator, lang='ja')}"
        except ValueError:
            return match.group(0)

    # 分数パターンの変換
    text = __FRACTION_PATTERN.sub(convert_fraction, text)

    # 単独の0時を零時に変換
    text = __ZERO_HOUR_PATTERN.sub(lambda m: f"{m.group(1) or ''}零時", text)

    # 時刻の処理（漢字で書かれた時分秒）
    def convert_time(match: re.Match[str]) -> str:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3)) if match.group(3) else None

        # 時刻として処理
        result = f"{num2words(hours, lang='ja')}時"

        # 分の処理：0分で秒がない場合は省略、秒がある場合は零分を追加
        if minutes == 0:
            if seconds is not None:
                result += "零分"
        elif 0 <= minutes <= 59:
            result += f"{num2words(minutes, lang='ja')}分"
        else:
            result += f"{num2words(minutes, lang='ja')}"

        # 秒の処理
        if seconds is not None:
            if 0 <= seconds <= 59:
                result += f"{num2words(seconds, lang='ja')}秒"
            else:
                result += f"{num2words(seconds, lang='ja')}"
        return result

    # 時刻パターンの処理（漢字で書かれた時分秒）
    text = __TIME_PATTERN.sub(convert_time, text)

    # 時刻またはアスペクト比の処理
    # 時刻は 00:00:00 から 27:59:59 までの範囲であれば、漢数字に変換して「十四時五分三十秒」「二十四時」のように読み上げる
    # それ以外ならアスペクト比と判断し「十六タイ九」のように読み上げる (「対」にすると「つい」と読んでしまう場合がある)
    def convert_time_or_aspect(match: re.Match[str]) -> str:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3)) if match.group(3) else None

        # 時刻として処理する条件をチェック
        # 時刻らしさを判定：時が0-27の範囲で、分が2桁で表現されている
        looks_like_time = 0 <= hours <= 27 and len(match.group(2)) == 2

        if looks_like_time:
            # 時刻として処理
            result = f"{num2words(hours, lang='ja')}時"

            # 分の処理：0分で秒がない場合は省略、秒がある場合は零分を追加
            if minutes == 0:
                if seconds is not None:
                    result += "零分"
            elif 0 <= minutes <= 59:
                result += f"{num2words(minutes, lang='ja')}分"
            else:
                result += f"{num2words(minutes, lang='ja')}"

            # 秒の処理
            if seconds is not None:
                if 0 <= seconds <= 59:
                    result += f"{num2words(seconds, lang='ja')}秒"
                else:
                    result += f"{num2words(seconds, lang='ja')}"
            return result
        else:
            # アスペクト比として処理
            result = f"{num2words(match.group(1), lang='ja')}タイ{num2words(match.group(2), lang='ja')}"
            if seconds is not None:
                result += f"タイ{num2words(seconds, lang='ja')}"
            return result

    # 時刻またはアスペクト比パターンの処理（コロンで区切られた時分秒）
    text = __ASPECT_PATTERN.sub(convert_time_or_aspect, text)

    # 指数表記の処理
    ## 稀にランダムな英数字 ID にマッチしたことで OverflowError が発生するが、続行に支障はないため無視する
    try:
        text = __EXPONENT_PATTERN.sub(
            lambda m: f"{num2words(float(m.group(0)), lang='ja')}", text
        )
    except OverflowError:
        pass

    # 記号類を辞書で置換
    text = __SYMBOL_YOMI_PATTERN.sub(lambda x: __SYMBOL_YOMI_MAP[x.group()], text)

    # 数字の前のバックスラッシュを円記号に変換
    ## __convert_numbers_to_words() は「¥100」を「100円」と自動で読み替えるが、円記号としてバックスラッシュ (U+005C) が使われているとうまく動作しないため
    ## ref: https://ja.wikipedia.org/wiki/%E5%86%86%E8%A8%98%E5%8F%B7
    text = re.sub(r"\\(?=\d)", "¥", text)

    return text


def __convert_numbers_to_words(text: str) -> str:
    """
    記号を日本語の文字表現に変換する。
    以前は数字を漢数字表現に変換していたが、pyopenjtalk 側の変換処理の方が優秀なため撤去した。

    Args:
        text (str): 変換するテキスト

    Returns:
        str: 変換されたテキスト
    """

    # 単位の変換（平方メートルなどの特殊な単位も含む）
    def convert_unit(match: re.Match[str]) -> str:
        number = match.group("number")
        unit = match.group("unit")
        suffix = match.group("suffix")
        # 特殊な単位の処理
        if unit.endswith("2"):
            base_unit = unit[:-1]
            if base_unit in __UNIT_MAP:
                return f"{number}平方{__UNIT_MAP[base_unit]}"
        elif unit.endswith("3"):
            base_unit = unit[:-1]
            if base_unit in __UNIT_MAP:
                return f"{number}立方{__UNIT_MAP[base_unit]}"
        # 指数表記の場合も単位変換を適用
        if "e" in str(number).lower():
            try:
                num_str = num2words(float(number), lang="ja")
                unit_str = __UNIT_MAP.get(unit, unit)
                return f"{num_str}{unit_str}"
            except (ValueError, OverflowError):
                pass
        # 単位が /s で終わるなら「毎秒」、/h で終わるなら「毎時」をつける
        if suffix == "/s":
            return f"{number}{__UNIT_MAP.get(unit, unit)}毎秒"
        elif suffix == "/h":
            return f"{number}{__UNIT_MAP.get(unit, unit)}毎時"
        else:
            return f"{number}{__UNIT_MAP.get(unit, unit)}"

    # 単位の変換
    res = __UNIT_PATTERN.sub(convert_unit, text)

    # 12,300 のような数字の区切りとしてのカンマを削除
    res = __NUMBER_WITH_SEPARATOR_PATTERN.sub(lambda m: m[0].replace(",", ""), res)

    # 通貨の変換
    res = __CURRENCY_PATTERN.sub(
        lambda m: (
            (m[2] + __CURRENCY_MAP.get(m[1], m[1]))
            if m[1]
            else (m[3] + __CURRENCY_MAP.get(m[4], m[4]))
        ),
        res,
    )

    return res


def __convert_english_to_katakana(text: str) -> str:
    """
    テキスト中の英単語をカタカナに変換する。
    複合語や略語、記号を含む単語など、様々なパターンに対応する。
    ただし、誤変換を防ぐため、確実に変換できるパターンのみを処理する。

    Args:
        text (str): 変換するテキスト

    Returns:
        str: 変換されたテキスト
    """

    def try_split_convert(word: str) -> str | None:
        """
        単語を2つに分割してカタカナ変換を試みる。
        中央から開始して左右に分割位置を移動しながら、両方の部分が辞書に存在する分割を探す。

        Args:
            word (str): 変換する単語

        Returns:
            str | None: 変換に成功した場合はカタカナ文字列、失敗した場合は None
        """

        # 単語を小文字に変換
        word = word.lower()
        n = len(word)

        # 分割位置の候補を生成（中央から左右に広がる順）
        center = n // 2
        # 中央から左右に移動する分割位置のリストを生成
        # 例: 長さ6の単語の場合、[3, 2, 4, 1, 5] の順で試す
        positions = []
        left = center
        right = center + 1
        while left > 0 or right < n:
            if left > 0:
                positions.append(left)
                left -= 1
            if right < n:
                positions.append(right)
                right += 1

        # 各分割位置で試行
        for pos in positions:
            part1 = word[:pos]
            part2 = word[pos:]

            # 両方の部分が辞書に存在するかチェック
            kata1 = KATAKANA_MAP.get(part1)
            if kata1 is None:
                continue

            kata2 = KATAKANA_MAP.get(part2)
            if kata2 is None:
                continue

            # 両方見つかった場合、カタカナを連結して返す
            # print(
            #     f"Split conversion succeeded: {word} -> {part1}({kata1}) + {part2}({kata2})"
            # )
            return kata1 + kata2

        return None

    def split_camel_case(word: str) -> list[str]:
        """
        CamelCase の単語を分割する。
        大文字が連続する場合はそれを一つの部分として扱う。

        Args:
            word (str): 分割する単語

        Returns:
            list[str]: 分割された部分文字列のリスト
        """

        parts = []
        current = word[0]
        prev_is_upper = word[0].isupper()

        for char in word[1:]:
            is_upper = char.isupper()

            # 小文字から大文字への変化、または大文字から小文字への変化を検出
            if (is_upper and not prev_is_upper) or (
                not is_upper and prev_is_upper and len(current) > 1
            ):
                parts.append(current)
                current = char
            else:
                current += char

            prev_is_upper = is_upper

        if current:
            parts.append(current)

        return parts

    def extract_alphabet_chunks(text: str) -> list[tuple[str, int, int]]:
        """
        テキストから連続するアルファベットのチャンクを抽出し、各チャンクとその開始・終了位置を返す。

        Args:
            text (str): 処理するテキスト

        Returns:
            list[tuple[str, int, int]]: (チャンク, 開始位置, 終了位置) のリスト
        """

        chunks = []
        current_chunk = ""
        start_pos = -1
        for i, char in enumerate(text):
            if __ALPHABET_PATTERN.match(char):
                if start_pos == -1:
                    start_pos = i
                current_chunk += char
            elif current_chunk:
                chunks.append((current_chunk, start_pos, i))
                current_chunk = ""
                start_pos = -1
        # 最後のチャンクを処理
        if current_chunk:
            chunks.append((current_chunk, start_pos, len(text)))
        return chunks

    def process_english_word(word: str, enable_romaji_c2k: bool = False) -> str:
        """
        英単語をカタカナに変換する。確実に変換できるパターンのみを処理し、
        不確実な場合は元の単語をそのまま返す (pyopenjtalk 側でアルファベット読みされる)。

        Args:
            word (str): 変換する英単語
            enable_romaji_c2k (bool): ローマ字変換や C2K によるカタカナ読みの推定を有効にするかどうか
        Returns:
            str: カタカナに変換された単語
        """

        # 事前に万が一 word の前後にスペースがあれば除去
        word = word.strip()
        # print(f"word: {word}")

        # 単体の大文字アルファベットは単位や記号として使われるケースが多く、
        # 安易にカタカナ読みすると不自然になりやすいため変換しない
        if len(word) == 1 and word.isupper() is True:
            return word

        # 数値（小数点を含む）を取り除いた後の文字列が UNIT_MAP に含まれる単位と完全一致する場合は実行しない
        # これにより、KATAKANA_MAP に "tb" が "ティービー" として別の読みで含まれていたとしても変換されずに済む
        word_without_numbers = __NUMBER_PATTERN.sub("", word)
        if word_without_numbers in __UNIT_MAP:
            return word

        # 英単語の末尾に 11 以下の数字 (1.0 のような小数表記を除く) がつく場合の処理 (例: iPhone 11, Pixel8)
        number_match = __ENGLISH_WORD_WITH_NUMBER_PATTERN.match(word)
        if number_match:
            base_word = number_match.group(1)
            number = number_match.group(2)
            # まず base_word をカタカナに変換できるか確認
            base_katakana = KATAKANA_MAP.get(base_word.lower())
            if base_katakana:
                # 数字を英語表現に変換し、それをカタカナに変換
                number_in_english = num2words(int(number), lang="en")
                number_katakana = process_english_word(
                    number_in_english, enable_romaji_c2k=True
                )
                if number_katakana:
                    return base_katakana + number_katakana

        # 1. 完全一致での変換を試みる（最も信頼できる変換）
        # 1.1 まず元の文字列で試す（辞書に大文字で登録されている頭字語はここで変換される）
        katakana_word = KATAKANA_MAP.get(word)
        if katakana_word:
            return katakana_word
        # 1.2 小文字に変換した上で試す
        katakana_word = KATAKANA_MAP.get(word.lower())
        if katakana_word:
            return katakana_word

        # 2. 末尾のピリオドを除去して再試行
        if word.endswith("."):
            katakana_word = KATAKANA_MAP.get(word[:-1].lower())
            if katakana_word:
                return katakana_word

        # 3. 所有格の処理（確実なパターン）
        if word.lower().endswith(("'s", "’s")):
            base_word = word[:-2]
            katakana_word = KATAKANA_MAP.get(base_word.lower())
            if katakana_word:
                return katakana_word + "ズ"

        # 4. 複数形の処理
        if word.endswith("s"):
            base_word = word[:-1]
            katakana_word = KATAKANA_MAP.get(base_word.lower())
            if katakana_word:
                return katakana_word + "ズ"

        # 5. 記号で区切られた複合語の処理（部分的な変換を許可）
        for separator, join_word in [
            ("&", "アンド"),
            ("-", ""),
            (".", ""),
            ("+", "プラス"),
        ]:
            if separator in word:
                # "." の場合は、小数点かどうかをチェック
                # "-" の場合は、数値の区切りかどうかをチェック
                if separator in [".", "-"]:
                    parts = word.split(separator)
                    # 隣接する部分が両方数字の場合は次のセパレータへ
                    should_skip = False
                    for i in range(len(parts) - 1):
                        if (
                            parts[i]
                            and parts[i][-1].isdigit()
                            and parts[i + 1]
                            and parts[i + 1][0].isdigit()
                        ):
                            should_skip = True
                            break
                    if should_skip:
                        continue

                sub_words = word.split(separator)
                katakana_sub_words = []

                for sub in sub_words:
                    # 大文字小文字に関わらず、分割した単語ごとに個別に変換を試みる
                    converted = process_english_word(sub, enable_romaji_c2k=True)
                    katakana_sub_words.append(converted)

                return join_word.join(katakana_sub_words)

        # 6. 数字（小数点含む）が含まれる場合、数字部分とそれ以外の部分に分割して処理
        if any(c.isdigit() for c in word):
            # ハイフンで区切られた数字の場合はそのまま返す (例: 33-4)
            if "-" in word:
                parts = word.split("-")
                if all(part.isdigit() for part in parts):
                    return word

            # "iPhone 11" "Pixel8" のようなパターンに一致しない場合のみ処理
            if not __ENGLISH_WORD_WITH_NUMBER_PATTERN.search(word):
                # 数字（小数点含む）とそれ以外の部分を分割
                parts = []
                last_end = 0

                for match in __NUMBER_PATTERN.finditer(word):
                    # 数字の前の部分を処理
                    if match.start() > last_end:
                        non_number = word[last_end : match.start()]
                        parts.append(
                            process_english_word(non_number, enable_romaji_c2k=True)
                        )

                    # 数字部分をそのまま追加
                    parts.append(match.group())
                    last_end = match.end()

                # 最後の非数字部分を処理
                if last_end < len(word):
                    non_number = word[last_end:]
                    parts.append(
                        process_english_word(non_number, enable_romaji_c2k=True)
                    )

                return "".join(parts)

        # CamelCase 分割がキャンセルされたかどうかを示すフラグ
        # CamelCase 分割がキャンセルされた場合は、2単語分割もスキップして直接 C2K に渡す
        # これにより、"KonoHa" のような単語が "kono" + "ha" に誤分割されるのを防ぐ
        camel_case_cancelled = False

        # 7. CamelCase の複合語を処理
        if any(c.isupper() for c in word[1:]):  # 2文字目以降に大文字が含まれる
            parts = split_camel_case(word)
            result_parts = []
            all_parts_converted = True  # すべてのパーツが変換できたかどうか

            for part in parts:
                # 3文字未満のアルファベットパーツは信頼性が低いため、分割をキャンセルする
                # 2文字以下の英単語（例: "Ha" → "ハー"）は複合語の一部として分割すると
                # 不自然な読みになることが多い（例: "KonoHa" → "コノハー" ではなく "コノハ" が正解）
                # ただし、数字のみで構成されるパーツ（例: "4", "3"）は長さチェックの対象外とする
                # （例: "GPT4Turbo" → "GPT" + "4" + "Turbo" の分割を許可する）
                if len(part) < 3 and not part.isdigit():
                    all_parts_converted = False
                    break

                # 数字のみで構成される部分はそのまま追加（後で pyopenjtalk で読まれる）
                if part.isdigit():
                    result_parts.append(part)
                # 大文字のみで構成される部分（例: "GPT", "API"）
                # 辞書検索時は小文字に変換して検索
                elif all(c.isupper() for c in part):
                    # まず大文字のまま検索、次に小文字で検索
                    converted = KATAKANA_MAP.get(part) or KATAKANA_MAP.get(part.lower())
                    if converted is not None:
                        result_parts.append(converted)
                    elif len(part) <= 4:
                        # 短い大文字パーツ（4文字以下）で辞書にない場合はそのまま追加
                        # （pyopenjtalk でアルファベット読みされる、例: "GPT" → "ジーピーティー"）
                        result_parts.append(part)
                    else:
                        # 長い大文字パーツで辞書にない場合は分割をキャンセルし、
                        # 元の単語全体を C2K で処理する（例: "WINDSURFEDITOR"）
                        all_parts_converted = False
                        break
                else:
                    # それ以外は辞書で変換を試みる
                    # enable_romaji_c2k を False に設定し、ローマ字変換、C2K 変換、2単語分割を無効にする
                    converted = process_english_word(part, enable_romaji_c2k=False)
                    # 変換結果が全てカタカナである場合のみ「変換成功」とみなす
                    # これにより、辞書にない単語（例: "Cono"）が混ざった場合は分割をキャンセルし、
                    # 元の単語全体を C2K に渡すことで正しい読みを取得できる
                    if is_all_katakana(converted):
                        result_parts.append(converted)
                    else:
                        all_parts_converted = False
                        break

            # すべてのパーツが変換できた場合のみ分割結果を使用
            if all_parts_converted is True:
                return "".join(result_parts)

            # 変換できなかったパーツがある場合はフラグを設定し、元の単語で fall through
            # 後続の2単語分割はスキップし、直接 C2K に渡す
            camel_case_cancelled = True

        # アルファベットのチャンクを抽出
        alpha_chunks = extract_alphabet_chunks(word)

        # 8. NGram モデルを用いて「英単語として読むか」「アルファベット読みするか」を判定する
        ## 「アルファベット読みと誤判定される可能性がある単語」の正しい読みが含まれることもあるので、
        ## 通常の変換処理を通した後に行っている
        ## オールキャップスな単語かつ、NGram モデルによってアルファベット読みすべきと判定された場合、そのままの表記で返す
        ## pyopenjtalk 側でアルファベット読みされるため、ここでカタカナに変換する必要はない
        if (
            word.isupper() is True
            and __should_transliterated_word_by_ngram(word) is False
        ):
            return word

        # 9. 最終手段として、2単語への分割を試みる
        # 最低4文字以上の単語のみ対象とし、全て大文字の単語の場合はこの処理を実行しない
        # enable_romaji_c2k が False の場合（CamelCase 分割後の再帰呼び出しなど）は、
        # 誤った分割を防ぐためこの処理をスキップする
        # CamelCase 分割がキャンセルされた場合も、誤分割を防ぐためこの処理をスキップする
        # （例: "KonoHa" が "kono" + "ha" に分割されて "コノハー" になるのを防ぐ）
        if (
            len(word) >= 4
            and not word.isupper()
            and enable_romaji_c2k is True
            and camel_case_cancelled is False
        ):
            split_result = try_split_convert(word)
            if split_result is not None:
                # 2単語分割の結果と C2K の結果を比較し、長音記号「ー」が少ない方を選ぶ
                # これにより、日本語のローマ字読み（例: "Musashi" → "ムサシ"）が
                # 英語の発音規則に基づいた辞書エントリ（例: "musa" → "ムーサ"）より優先される
                c2k_result = __characters_to_katakana(word.lower())
                split_long_vowel_count = split_result.count("ー")
                c2k_long_vowel_count = c2k_result.count("ー")
                if c2k_long_vowel_count < split_long_vowel_count:
                    return c2k_result
                return split_result

        # 10. 本当に最後の手段として、C2K によるカタカナ読みの推定を試みる
        ## この処理はあくまで辞書ベースで解決できなかった場合の最終手段なので、CamelCase を分割して個々の単語ごとに処理する際はこの処理は通らない
        ## 従来はこの処理を通す前に romkan でローマ字変換を通していたが、
        ## C2K が実質的にローマ字変換も行えるようになったので、現在はローマ字変換も C2K に任せている
        ## ref: https://github.com/Patchethium/e2k
        if alpha_chunks and enable_romaji_c2k is True:
            # 英単語の末尾に 11 以下の数字 (1.0 のような小数表記を除く) がつく場合の処理 (例: iPhone 11, Pixel8)
            number_match = __ENGLISH_WORD_WITH_NUMBER_PATTERN.match(word)
            if number_match:
                base_word = number_match.group(1)
                number = number_match.group(2)
                # まず base_word をカタカナに変換
                # c2k は小文字でのみ動作する
                converted_katakana = __characters_to_katakana(base_word.lower())
                # 数字を英語表現に変換し、それをカタカナに変換
                number_in_english = num2words(int(number), lang="en")
                number_katakana = process_english_word(
                    number_in_english, enable_romaji_c2k=True
                )
                if number_katakana:
                    return converted_katakana + number_katakana

            # 変換情報を保存するリスト
            replacements = []
            converted_any = False

            # 各チャンクに対して処理10を実行
            for chunk, start, end in alpha_chunks:
                # オールキャップスではない単語では常に c2k を通す
                # オールキャップスな単語だが、NGram によって英単語として読むべきと判定された場合は c2k を通す
                # いずれも3文字以上の場合のみ実行する (dB など単位系の誤変換を避けるため)
                if len(chunk) >= 3 and (
                    chunk.isupper() is False
                    or __should_transliterated_word_by_ngram(chunk) is True
                ):
                    # いずれかの文字がアルファベットの場合のみ適用
                    if any(__ALPHABET_PATTERN.match(c) for c in chunk):
                        # c2k は小文字でのみ動作する
                        converted = __characters_to_katakana(chunk.lower())
                        converted_any = True
                        replacements.append((start, end, converted))

            # 置換情報を元に新しい文字列を構築（後ろから処理することで位置ずれを防ぐ）
            if converted_any:
                for start, end, converted in sorted(replacements, reverse=True):
                    # start と end が単語の長さ内にあることを確認
                    if start < len(word) and end <= len(word):
                        # 元の単語のチャンク部分を変換結果で置き換える
                        word = word[:start] + converted + word[end:]

        # 上記以外は元の単語を返す (pyopenjtalk 側でアルファベット読みされる)
        return word

    def is_all_katakana(s: str) -> bool:
        """
        文字列が全てカタカナで構成されているかどうかを判定する。

        Args:
            s (str): 判定する文字列

        Returns:
            bool: 全てカタカナで構成されている場合は True、そうでない場合は False
        """

        # 空文字列の場合はFalseを返す
        if not s:
            return False

        # Unicode のカタカナブロックは U+30A0 ~ U+30FF
        for c in s:
            if not ("\u30a0" <= c <= "\u30ff"):
                return False
        return True

    # NFKC 処理でいくつかハイフンの変種が U+002D とは別のハイフンである U+2010 に変換されるので、それを通常のハイフンに変換する
    text = text.replace("\u2010", "-")

    # 単語中で使われうるクオートを全て ' に置換する (例: We’ve -> We've)
    quotes = [
        "\u2018",  # LEFT SINGLE QUOTATION MARK ‘
        "\u2019",  # RIGHT SINGLE QUOTATION MARK ’
        "\u201a",  # SINGLE LOW-9 QUOTATION MARK ‚
        "\u201b",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK ‛
        "\u2032",  # PRIME ′
        "\u0060",  # GRAVE ACCENT `
        "\u00b4",  # ACUTE ACCENT ´
        "\u2033",  # DOUBLE PRIME ″
        "\u301d",  # REVERSED DOUBLE PRIME QUOTATION MARK 〝
        "\u301e",  # DOUBLE PRIME QUOTATION MARK 〞
        "\u301f",  # LOW DOUBLE PRIME QUOTATION MARK 〟
        "\uff07",  # FULLWIDTH APOSTROPHE ＇
    ]
    # 全てのクオート記号を ' に置換
    for quote in quotes:
        text = text.replace(quote, "'")

    words = []
    current_word = ""
    prev_char = ""
    # 英単語がカタカナに変換されたかどうかを記録するフラグのリスト
    is_english_converted = []

    # 敬称のパターンを定義（ピリオド付きと無しの両方）
    title_patterns = [
        (r"Mrs\.?", "ミセス"),
        (r"Mr\.?", "ミスター"),
        (r"Ms\.?", "ミズ"),
        (r"Mx\.?", "ミクス"),
        (r"Dr\.?", "ドクター"),
        (r"Esq\.?", "エスク"),
        (r"Jr\.?", "ジュニア"),
        (r"Sr\.?", "シニア"),
    ]

    i = 0
    while i < len(text):
        # 敬称パターンのマッチを試みる
        matched = False
        for pattern, replacement in title_patterns:
            match = re.match(pattern, text[i:])
            if match:
                words.append(replacement)
                is_english_converted.append(True)  # 敬称は変換されたものとして記録
                i += len(match.group(0))
                matched = True
                current_word = ""
                break

        if matched:
            continue

        char = text[i]
        next_char = text[i + 1] if i < len(text) - 1 else ""

        # 英単語の後に0-11の数字が続く場合の特別処理
        if current_word and __ALPHABET_PATTERN.search(current_word) and char.isdigit():
            # 現在位置から数字を抽出（小数点も含めて）
            num_str = ""
            j = i
            has_decimal_point = False

            # 数字部分を抽出（小数点も含む）
            while j < len(text) and (text[j].isdigit() or text[j] == "."):
                if text[j] == ".":
                    has_decimal_point = True
                num_str += text[j]
                j += 1

            # 小数点を含む場合はこの特別処理を行わない
            if has_decimal_point:
                current_word += num_str
                i = j  # 数字（小数点含む）の最後の位置まで進める
                continue

            # 整数部分のみを取得（小数点がない場合は元の数字文字列と同じ）
            int_part = num_str

            # 0-11の数字であり、かつその後に英数字が続く場合は分割
            if int_part and 0 <= int(int_part) <= 11:
                # 数字の後に英数字が続くかどうかを確認
                has_alnum_after = j < len(text) and (
                    __ENGLISH_WORD_PATTERN.match(text[j]) is not None
                )

                if has_alnum_after:
                    # 英単語を処理
                    is_all_alpha = all(
                        __ALPHABET_PATTERN.match(c) for c in current_word
                    )
                    converted = process_english_word(
                        current_word, enable_romaji_c2k=True
                    )
                    words.append(converted)
                    is_english_converted.append(
                        is_all_alpha
                        and (is_all_katakana(converted) or converted.isupper())
                    )
                    current_word = int_part  # 数字を新しい単語として設定
                    i = j  # 数字の最後の位置まで進める
                    continue
                else:
                    # 英単語+数字を一つの単語として扱う
                    current_word += int_part
                    i = j  # 数字の最後の位置まで進める
                    continue

        # 英数字または特定の記号であれば current_word に追加
        if __ENGLISH_WORD_PATTERN.match(char) is not None or char in "-&+'":
            current_word += char
        # ピリオドの特別処理
        elif char == ".":
            # 前後が英数字の場合は単語の一部として扱う (例: Node.js)
            if (
                current_word
                and next_char
                and (
                    __ENGLISH_WORD_PATTERN.match(prev_char) is not None
                    and __ENGLISH_WORD_PATTERN.match(next_char) is not None
                )
            ):
                current_word += char
            # それ以外は文の区切りとして扱う (例: I'm fine.)
            else:
                if current_word:
                    # 元の単語が全てアルファベットかどうかを確認
                    is_all_alpha = all(
                        __ALPHABET_PATTERN.match(c) for c in current_word
                    )
                    # 変換処理
                    converted = process_english_word(
                        current_word, enable_romaji_c2k=True
                    )
                    words.append(converted)
                    # 変換後が全てカタカナかつ元が全てアルファベットなら True
                    is_english_converted.append(
                        is_all_alpha and is_all_katakana(converted)
                    )
                    current_word = ""
                words.append(char)
                is_english_converted.append(False)  # 記号は変換されていない
        # スペースまたはハイフンの特別処理（英単語の後に0-11の数字が続く場合）
        elif (
            (char == " " or char == "-")
            and current_word
            and __ALPHABET_PATTERN.search(current_word)
        ):
            # 次の文字が0-11の数字かどうかを確認
            if next_char.isdigit():
                # 0-11の数字を抽出
                num_str = ""
                j = i + 1
                while j < len(text) and text[j].isdigit():
                    num_str += text[j]
                    j += 1
                # 0-11の数字であり、かつその後に英数字が続かない場合
                if num_str and 0 <= int(num_str) <= 11:
                    # 数字の後に英数字が続くかどうかを確認
                    has_alnum_after = j < len(text) and (
                        __ENGLISH_WORD_PATTERN.match(text[j]) is not None
                    )
                    if not has_alnum_after:
                        # 英単語+スペース/ハイフン+数字を一つの単語として扱う
                        current_word += char + num_str
                        i = j  # 数字の最後の位置まで進める
                        continue
            # 上記条件に当てはまらない場合は通常処理
            if current_word:
                # 元の単語が全てアルファベットかどうかを確認
                is_all_alpha = all(__ALPHABET_PATTERN.match(c) for c in current_word)
                # 変換処理
                converted = process_english_word(current_word, enable_romaji_c2k=True)
                words.append(converted)
                # 変換後が全てカタカナかつ元が全てアルファベットなら、もしくは当該単語が全て大文字からなる場合は True
                is_english_converted.append(
                    is_all_alpha and (is_all_katakana(converted) or converted.isupper())
                )
                current_word = ""
            words.append(char)
            is_english_converted.append(False)  # スペースやハイフンは変換されていない
        else:
            # 英単語が終了したらカタカナに変換して words に追加
            if current_word:
                # 元の単語が全てアルファベットかどうかを確認
                is_all_alpha = all(__ALPHABET_PATTERN.match(c) for c in current_word)
                # 変換処理
                converted = process_english_word(current_word, enable_romaji_c2k=True)
                words.append(converted)
                # 変換後が全てカタカナかつ元が全てアルファベットなら、もしくは当該単語が全て大文字からなる場合は True
                is_english_converted.append(
                    is_all_alpha and (is_all_katakana(converted) or converted.isupper())
                )
                current_word = ""
            words.append(char)
            is_english_converted.append(False)  # 記号や他の文字は変換されていない

        prev_char = char
        i += 1

    # 最後の単語を処理
    if current_word:
        # 元の単語が全てアルファベットかどうかを確認
        is_all_alpha = all(__ALPHABET_PATTERN.match(c) for c in current_word)
        # 変換処理
        converted = process_english_word(current_word, enable_romaji_c2k=True)
        words.append(converted)
        # 変換後が全てカタカナかつ元が全てアルファベットなら True
        is_english_converted.append(is_all_alpha and is_all_katakana(converted))

    # 単数を表す "a" の処理
    # 「a」の直後に空白があり、その後の単語が英語からカタカナに変換されている場合、「ア」に置き換える
    new_words = []
    i = 0
    while i < len(words):
        # "a" が現れたら、後続に空白をスキップして次のトークンを取得
        if words[i] == "a":
            j = i + 1
            # j 以降が空白ならスキップ
            while j < len(words) and words[j].isspace():
                j += 1
            # 次のトークンが英語からカタカナに変換されたものかどうか確認
            if (
                j < len(words)
                and j < len(is_english_converted)
                and is_english_converted[j]
            ):
                # "a" を「ア」として、後続の英単語と結合
                new_words.append("ア" + words[j])
                i = j + 1  # 置換済みなのでスキップ
                continue
        # その他はそのまま追加
        new_words.append(words[i])
        i += 1

    return "".join(new_words)


def replace_punctuation(text: str) -> str:
    """
    句読点等を「.」「,」「!」「?」「'」「-」に正規化し、OpenJTalk で読みが取得できるもののみ残す：
    漢字・平仮名・カタカナ、数字、アルファベット、ギリシャ文字

    Args:
        text (str): 正規化するテキスト

    Returns:
        str: 正規化されたテキスト
    """

    # 句読点を辞書で置換
    replaced_text = __SYMBOL_REPLACE_PATTERN.sub(
        lambda x: __SYMBOL_REPLACE_MAP[x.group()], text
    )

    # 上述以外の文字を削除
    replaced_text = __PUNCTUATION_CLEANUP_PATTERN.sub("", replaced_text)

    return replaced_text


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m style_bert_vits2.nlp.japanese.normalizer <text>")
        sys.exit(1)
    print(normalize_text(sys.argv[1]))
