from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

LOADED_MODELS = {}

# Exact single-word fallback overrides for high-frequency edge cases
COMMON_OVERRIDES = {
    ("de", "en"): {
        "nein": "No",
        "ja": "Yes",
        "danke": "Thank you",
        "bitte": "Please / You're welcome",
    },
    ("en", "de"): {
        "no": "Nein",
        "yes": "Ja",
        "thank you": "Danke",
        "thanks": "Danke",
    },
}

LANG_CONFIG = {
    "de": {"token": ">>deu<<"},
    "en": {"token": ">>eng<<"},
    "es": {"token": ">>spa<<"},
    "fr": {"token": ">>fra<<"},
    "it": {"token": ">>ita<<"},
    "pt": {"token": ">>por<<"},
    "zh": {"token": ">>zho<<"},
    "ja": {"token": ">>jap<<"},
    "hi": {"token": ">>hin<<"},
    "ml": {"token": ">>mal<<"},
    "ta": {"token": ">>tam<<"},
    "te": {"token": ">>tel<<"},
    "kn": {"token": ">>kan<<"},
}

def get_hf_model_names(src: str, tgt: str):
    """Generate prioritized Hugging Face model repository IDs."""
    candidates = [
        f"Helsinki-NLP/opus-mt-{src}-{tgt}",
        f"Helsinki-NLP/opus-mt-tc-big-{src}-{tgt}",
    ]
    
    if src == "en" and tgt in ["pt", "es", "fr", "it"]:
        candidates.append("Helsinki-NLP/opus-mt-en-ROMANCE")
    elif src in ["pt", "es", "fr", "it"] and tgt == "en":
        candidates.append("Helsinki-NLP/opus-mt-ROMANCE-en")
        
    if tgt == "en":
        candidates.append("Helsinki-NLP/opus-mt-mul-en")
    elif src == "en":
        candidates.append("Helsinki-NLP/opus-mt-en-mul")

    return candidates

def load_model_pair(src: str, tgt: str):
    """Load and cache tokenizer + model."""
    for model_name in get_hf_model_names(src, tgt):
        if model_name in LOADED_MODELS:
            return LOADED_MODELS[model_name]
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            LOADED_MODELS[model_name] = (tokenizer, model)
            return tokenizer, model
        except Exception:
            continue
    return None, None

def run_inference(text: str, src: str, tgt: str, tokenizer, model) -> str:
    """Run inference with adaptive decoding parameters."""
    tgt_token = LANG_CONFIG.get(tgt, {}).get("token", "")
    model_path = getattr(model.config, "_name_or_path", "")
    
    # Pre-format target tokens for multi-target or romance models
    if ("ROMANCE" in model_path or "-mul" in model_path) and tgt_token:
        formatted_text = f"{tgt_token} {text}"
    else:
        formatted_text = text

    inputs = tokenizer(formatted_text, return_tensors="pt", padding=True)
    
    word_count = len(text.split())
    
    # For single words / short phrases, use greedy search (num_beams=1) to prevent double outputs like "no no"
    if word_count <= 3:
        generation_kwargs = {
            "max_length": 128,
            "num_beams": 1,
            "do_sample": False,
        }
    else:
        generation_kwargs = {
            "max_length": 512,
            "num_beams": 4,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
        }

    translated_tokens = model.generate(**inputs, **generation_kwargs)
    
    decoded_text = tokenizer.decode(
        translated_tokens[0], 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).strip()
    
    return decoded_text

def translate_text_pipeline(text: str, source_lang: str, target_lang: str) -> str:
    """Try direct translation with overrides; if missing, pivot through English."""
    normalized_input = text.strip().lower().rstrip(".!?")
    
    # Check exact dictionary override first
    if (source_lang, target_lang) in COMMON_OVERRIDES:
        if normalized_input in COMMON_OVERRIDES[(source_lang, target_lang)]:
            return COMMON_OVERRIDES[(source_lang, target_lang)][normalized_input]

    # Step 1: Direct Translation
    tok, mod = load_model_pair(source_lang, target_lang)
    if tok and mod:
        return run_inference(text, source_lang, target_lang, tok, mod)

    # Step 2: English Pivot (Source -> EN -> Target)
    if source_lang != "en" and target_lang != "en":
        tok_src_en, mod_src_en = load_model_pair(source_lang, "en")
        tok_en_tgt, mod_en_tgt = load_model_pair("en", target_lang)

        if tok_src_en and mod_src_en and tok_en_tgt and mod_en_tgt:
            intermediate_en = run_inference(text, source_lang, "en", tok_src_en, mod_src_en)
            return run_inference(intermediate_en, "en", target_lang, tok_en_tgt, mod_en_tgt)

    raise ValueError(f"Unable to resolve translation model for {source_lang.upper()} → {target_lang.upper()}")

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
        translated_text = translate_text_pipeline(clean_text, source_lang, target_lang)
        return f'<p class="text-slate-100 font-medium text-sm leading-relaxed">{translated_text}</p>'
    except Exception as e:
        return f'<p class="text-amber-400 font-medium text-sm">Translation error: {str(e)}</p>'