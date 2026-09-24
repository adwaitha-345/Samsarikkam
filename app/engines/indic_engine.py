import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Updated with correct FLORES-200 3-letter codes
NLLB_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ml": "mal_Mlym",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
}

# Unicode Script range checking to prevent language mismatch
SCRIPT_RANGES = {
    "hi": [("\u0900", "\u097F")],  # Devanagari
    "ml": [("\u0D00", "\u0D7F")],  # Malayalam
    "ta": [("\u0B80", "\u0BFF")],  # Tamil
    "te": [("\u0C00", "\u0C7F")],  # Telugu
    "kn": [("\u0C80", "\u0CFF")],  # Kannada
}

print("Loading NLLB Indic Engine (facebook/nllb-200-distilled-600M)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
print("NLLB Indic Engine loaded successfully!")

def matches_expected_script(text: str, lang_code: str) -> bool:
    """Verifies if input text matches the script of the declared source language."""
    if lang_code not in SCRIPT_RANGES:
        return True  # Skip check for Latin / non-Indic scripts
    
    ranges = SCRIPT_RANGES[lang_code]
    for char in text:
        if char.isspace() or char.isdigit() or char in ".,!?-()":
            continue
        # Check if the character falls into expected unicode range
        if any(start <= char <= end for start, end in ranges):
            return True
    return False

def translate_indic(text: str, source_lang: str, target_lang: str) -> str:
    # Script validation check
    if source_lang in SCRIPT_RANGES and not matches_expected_script(text, source_lang):
        raise ValueError(f"Input text does not match the selected source language script ({source_lang.upper()}).")

    src_code = NLLB_CODES.get(source_lang, "eng_Latn")
    tgt_code = NLLB_CODES.get(target_lang, "eng_Latn")

    # Set source language on tokenizer
    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt")

    # Retrieve target language token ID
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_code)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=512
        )

    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return result[0]