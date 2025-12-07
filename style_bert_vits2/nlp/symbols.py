# Punctuations
PUNCTUATIONS = ["!", "?", "…", ",", ".", "'", "-"]

# Punctuations and special tokens
PUNCTUATION_SYMBOLS = PUNCTUATIONS + ["SP", "UNK"]

# Padding
PAD = "_"

# ============================================================================
# Legacy symbols (v2) - kept for backwards compatibility with pre-3.x models
# ============================================================================

# Chinese symbols (legacy, kept for v2 model loading)
_ZH_SYMBOLS = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
_NUM_ZH_TONES = 6

# English symbols (legacy, kept for v2 model loading)
_EN_SYMBOLS = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
_NUM_EN_TONES = 4

# ============================================================================
# Japanese symbols (current, used in v3)
# ============================================================================

JP_SYMBOLS = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "v",
    "w",
    "y",
    "z",
    "zy",
]
NUM_JP_TONES = 2

# ============================================================================
# Symbol versioning for model compatibility
# ============================================================================

# v2 symbols (legacy, for pre-3.x models)
# Total: 1 (PAD) + ~98 unique symbols + 9 punctuation = ~108
_NORMAL_SYMBOLS_V2 = sorted(set(_ZH_SYMBOLS + JP_SYMBOLS + _EN_SYMBOLS))
SYMBOLS_V2 = [PAD] + _NORMAL_SYMBOLS_V2 + PUNCTUATION_SYMBOLS
NUM_TONES_V2 = _NUM_ZH_TONES + NUM_JP_TONES + _NUM_EN_TONES  # 12

# v2 language/tone mappings (legacy)
# In v2: ZH=0, JP=1, EN=2 for language IDs
# In v2: ZH tones 0-5, JP tones 6-7, EN tones 8-11
LANGUAGE_ID_MAP_V2 = {"ZH": 0, "JP": 1, "EN": 2}
LANGUAGE_TONE_START_MAP_V2 = {"ZH": 0, "JP": 6, "EN": 8}

# v3 symbols (current, JP-only)
# Total: 1 (PAD) + 43 JP symbols + 9 punctuation = 53
_NORMAL_SYMBOLS_V3 = sorted(set(JP_SYMBOLS))
SYMBOLS_V3 = [PAD] + _NORMAL_SYMBOLS_V3 + PUNCTUATION_SYMBOLS
NUM_TONES_V3 = NUM_JP_TONES  # 2

# ============================================================================
# Current exports (v3 by default)
# ============================================================================

# Default to v3 for new code
SYMBOLS = SYMBOLS_V3
NORMAL_SYMBOLS = _NORMAL_SYMBOLS_V3
SIL_PHONEMES_IDS = [SYMBOLS.index(i) for i in PUNCTUATION_SYMBOLS]

# Tones (JP-only in v3)
NUM_TONES = NUM_TONES_V3

# Language maps (JP-only in v3)
LANGUAGE_ID_MAP = {"JP": 0}
NUM_LANGUAGES = 1

# Language tone start map (JP-only, starts at 0)
LANGUAGE_TONE_START_MAP = {"JP": 0}

# ============================================================================
# Legacy exports for backwards compatibility
# ============================================================================

# These are kept for any code that might still reference them
# but they should not be used in new code
ZH_SYMBOLS = _ZH_SYMBOLS  # Deprecated
EN_SYMBOLS = _EN_SYMBOLS  # Deprecated
NUM_ZH_TONES = _NUM_ZH_TONES  # Deprecated
NUM_EN_TONES = _NUM_EN_TONES  # Deprecated


if __name__ == "__main__":
    print(f"SYMBOLS_V2 count: {len(SYMBOLS_V2)}")
    print(f"SYMBOLS_V3 count: {len(SYMBOLS_V3)}")
    print(f"NUM_TONES_V2: {NUM_TONES_V2}")
    print(f"NUM_TONES_V3: {NUM_TONES_V3}")
    print(f"\nSYMBOLS_V3: {SYMBOLS_V3}")
