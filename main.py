import re
import argostranslate.package
import argostranslate.translate
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

SUPPORTED_LANGUAGES = ["en", "es", "de", "fr", "it", "pt", "zh", "ja", "hi", "ml", "ta", "te", "kn"]

# Lookup table for common English single words to prevent model decoding failures
DIRECT_LOOKUP = {
    ("en", "ja"): {
        "hello": "こんにちは",
        "hi": "こんにちは",
        "thank you": "ありがとうございます",
        "thanks": "ありがとう",
        "yes": "はい",
        "no": "いいえ",
        "good morning": "おはようございます",
        "good evening": "こんばんは",
        "goodbye": "さようなら",
        "bye": "じゃあね",
        "please": "お願いします",
        "welcome": "ようこそ",
        "sorry": "ごめんなさい",
    },
    ("ja", "en"): {
        "こんにちは": "Hello",
        "ありがとう": "Thank you",
        "ありがとうございます": "Thank you very much",
        "はい": "Yes",
        "いいえ": "No",
        "おはようございます": "Good morning",
        "こんばんは": "Good evening",
        "さようなら": "Goodbye",
    },
    ("hi", "en"): {
        "नमस्ते": "Hello",
        "नमस्कार": "Hello",
        "धन्यवाद": "Thank you",
        "हाँ": "Yes",
        "नहीं": "No",
    },
    ("ml", "en"): {
        "നമസ്കാരം": "Hello",
        "നന്ദി": "Thank you",
        "അതെ": "Yes",
        "ഇല്ല": "No",
    },
}

def setup_argos_models():
    """Download and install available translation packages from Argos index."""
    print("Checking and updating Argos Translate package index...")
    try:
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        installed_packages = argostranslate.package.get_installed_packages()
        
        installed_pairs = {
            (pkg.from_code, pkg.to_code) for pkg in installed_packages
        }

        for pkg in available_packages:
            if pkg.from_code in SUPPORTED_LANGUAGES and pkg.to_code in SUPPORTED_LANGUAGES:
                if (pkg.from_code, pkg.to_code) not in installed_pairs:
                    print(f"Downloading translation package: {pkg.from_code} -> {pkg.to_code}...")
                    download_path = pkg.download()
                    argostranslate.package.install_from_path(download_path)
                    
        print("Argos Translation engine ready!")
    except Exception as e:
        print(f"Package initialization note: {e}")

setup_argos_models()

def contains_japanese(text: str) -> bool:
    """Check if text contains Japanese Hiragana, Katakana, or Kanji characters."""
    return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))

def translate_text_argos(text: str, source_lang: str, target_lang: str) -> str:
    clean_text = text.strip()
    normalized_text = clean_text.lower().rstrip(".!?")

    # 1. Direct Lookup Check for English-to-Japanese and other exact pairs
    if (source_lang, target_lang) in DIRECT_LOOKUP:
        if clean_text in DIRECT_LOOKUP[(source_lang, target_lang)]:
            return DIRECT_LOOKUP[(source_lang, target_lang)][clean_text]
        if normalized_text in DIRECT_LOOKUP[(source_lang, target_lang)]:
            return DIRECT_LOOKUP[(source_lang, target_lang)][normalized_text]

    # 2. First Pass: Standard Argos Inference
    result = argostranslate.translate.translate(clean_text, source_lang, target_lang).strip()

    # 3. Japanese Specific Handling
    if target_lang == "ja":
        # If the result doesn't contain Japanese script (e.g. model outputted English back), retry with context
        if not contains_japanese(result):
            contextual_input = f"The translation is: {clean_text}."
            retried = argostranslate.translate.translate(contextual_input, source_lang, target_lang).strip()
            # Clean up introductory translated phrases if present
            cleaned_retry = re.sub(r'^(翻訳は|訳：| translation is:?\s*)', '', retried).strip()
            if contains_japanese(cleaned_retry):
                return cleaned_retry

    elif source_lang == "ja" and target_lang == "en":
        # If target is English but Japanese characters remain in output
        if contains_japanese(result):
            contextual_input = f"「{clean_text}」"
            retried = argostranslate.translate.translate(contextual_input, source_lang, target_lang).strip()
            cleaned_retry = re.sub(r'[「」"\'”]', '', retried).strip()
            if cleaned_retry and not contains_japanese(cleaned_retry):
                return cleaned_retry

    return result

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/translate", response_class=HTMLResponse)
async def translate_text(
    request: Request, 
    text: str = Form(""),
    source_lang: str = Form(""),
    target_lang: str = Form("")
):
    clean_text = text.strip()
    if not clean_text:
        return '<p class="italic text-slate-500">Please enter text to translate.</p>'
    
    if not source_lang or not target_lang:
        return '<p class="text-amber-400 font-medium text-sm">Please select both source and target languages.</p>'

    if source_lang == target_lang:
        return f'<p class="text-slate-100 font-medium text-sm leading-relaxed">{clean_text}</p>'

    try:
        translated_text = translate_text_argos(clean_text, source_lang, target_lang)
        if not translated_text:
            return '<p class="text-amber-400 font-medium text-sm">Could not translate text. Please verify language selection.</p>'
            
        return f'<p class="text-slate-100 font-medium text-sm leading-relaxed">{translated_text}</p>'
    except Exception as e:
        return f'<p class="text-amber-400 font-medium text-sm">Translation error: {str(e)}</p>'