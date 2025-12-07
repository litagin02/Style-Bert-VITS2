from style_bert_vits2.constants import VERSION, Languages
from style_bert_vits2.nlp.japanese import g2p
from style_bert_vits2.tts_model import TTSModelHolder


def test_library_imports():
    """
    Verify that key components can be imported as a library.
    This ensures the package structure allows users to access the main classes and functions.
    """
    # Ensure version is present
    assert VERSION is not None

    # Ensure the main class is importable
    assert TTSModelHolder is not None

    # Ensure NLP components are reachable
    assert g2p is not None

    # Ensure Enums are working
    assert Languages.JP == "JP"

    print(f"Library version: {VERSION}")
    print("Successfully imported TTSModelHolder, g2p, and Languages.")
