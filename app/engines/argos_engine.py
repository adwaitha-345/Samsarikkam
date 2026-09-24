import re
import argostranslate.package
import argostranslate.translate

ARGOS_LANGUAGES = {"en", "es", "de", "fr", "it", "pt", "zh", "ja"}

DIRECT_LOOKUP = {
    ("en", "ja"): {"hello": "こんにちは", "thank you": "ありがとうございます", "yes": "はい", "no": "いいえ"},
    ("ja", "en"): {"こんにちは": "Hello", "ありがとう": "Thank you"},
}

def setup_argos_models():
    """Download and install available translation packages for Argos languages."""
    print("Initializing Argos Translate engine...")
    try:
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        installed_packages = argostranslate.package.get_installed_packages()
        
        installed_pairs = {(pkg.from_code, pkg.to_code) for pkg in installed_packages}

        for pkg in available_packages:
            if pkg.from_code in ARGOS_LANGUAGES and pkg.to_code in ARGOS_LANGUAGES:
                if (pkg.from_code, pkg.to_code) not in installed_pairs:
                    print(f"Downloading Argos package: {pkg.from_code} -> {pkg.to_code}...")
                    download_path = pkg.download()
                    argostranslate.package.install_from_path(download_path)
                    
        print("Argos Engine initialized successfully!")
    except Exception as e:
        print(f"Argos setup note: {e}")

# Initialize Argos packages on startup
setup_argos_models()

def contains_japanese(text: str) -> bool:
    return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))

def translate_argos(text: str, source_lang: str, target_lang: str) -> str:
    clean_text = text.strip()
    normalized_text = clean_text.lower().rstrip(".!?")

    if (source_lang, target_lang) in DIRECT_LOOKUP:
        if clean_text in DIRECT_LOOKUP[(source_lang, target_lang)]:
            return DIRECT_LOOKUP[(source_lang, target_lang)][clean_text]
        if normalized_text in DIRECT_LOOKUP[(source_lang, target_lang)]:
            return DIRECT_LOOKUP[(source_lang, target_lang)][normalized_text]

    result = argostranslate.translate.translate(clean_text, source_lang, target_lang).strip()

    if target_lang == "ja" and not contains_japanese(result):
        retried = argostranslate.translate.translate(f"The translation is: {clean_text}.", source_lang, target_lang).strip()
        cleaned = re.sub(r'^(翻訳は|訳：| translation is:?\s*)', '', retried).strip()
        if contains_japanese(cleaned):
            return cleaned

    return result