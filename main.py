from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

LOADED_MODELS = {}

# Map standard codes to HF naming & tokens
LANG_CONFIG = {
    "ja": {"hf_code": "jap", "token": ">>jap<<"},
    "pt": {"hf_code": "pt", "token": ">>por<<"},
    "es": {"hf_code": "es", "token": ">>spa<<"},
    "fr": {"hf_code": "fr", "token": ">>fra<<"},
    "it": {"hf_code": "it", "token": ">>ita<<"},
    "de": {"hf_code": "de", "token": ">>deu<<"},
    "en": {"hf_code": "en", "token": ">>eng<<"},
    "zh": {"hf_code": "zh", "token": ">>zho<<"},
    "hi": {"hf_code": "hi", "token": ">>hin<<"},
    "ml": {"hf_code": "ml", "token": ">>mal<<"},
    "ta": {"hf_code": "ta", "token": ">>tam<<"},
    "te": {"hf_code": "te", "token": ">>tel<<"},
    "kn": {"hf_code": "kn", "token": ">>kan<<"},
}

def get_hf_model_names(src: str, tgt: str):
    """Generate candidate Hugging Face model repository IDs for a pair."""
    src_cfg = LANG_CONFIG.get(src, {"hf_code": src})
    tgt_cfg = LANG_CONFIG.get(tgt, {"hf_code": tgt})
    
    s_code, t_code = src_cfg["hf_code"], tgt_cfg["hf_code"]
    
    candidates = [
        f"Helsinki-NLP/opus-mt-{s_code}-{t_code}",
        f"Helsinki-NLP/opus-mt-tc-big-{s_code}-{t_code}",
    ]
    
    # Romance group fallbacks
    if src == "en" and tgt in ["pt", "es", "fr", "it"]:
        candidates.append("Helsinki-NLP/opus-mt-en-ROMANCE")
    elif src in ["pt", "es", "fr", "it"] and tgt == "en":
        candidates.append("Helsinki-NLP/opus-mt-ROMANCE-en")
        
    # Multilingual fallbacks for languages without dedicated single-pair repos
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
    """Run inference with required target tokens."""
    tgt_token = LANG_CONFIG.get(tgt, {}).get("token", "")
    
    # Check if target token is needed for multi-target or romance models
    model_path = getattr(model.config, "_name_or_path", "")
    if ("ROMANCE" in model_path or "-mul" in model_path) and tgt_token:
        formatted_text = f"{tgt_token} {text}"
    else:
        formatted_text = text

    inputs = tokenizer(formatted_text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def translate_text_pipeline(text: str, source_lang: str, target_lang: str) -> str:
    """Try direct translation; if missing, pivot through English."""
    # Step 1: Direct Translation Attempt
    tok, mod = load_model_pair(source_lang, target_lang)
    if tok and mod:
        return run_inference(text, source_lang, target_lang, tok, mod)

    # Step 2: English Pivot (Source -> EN -> Target)
    if source_lang != "en" and target_lang != "en":
        # Step A: Source -> EN
        tok_src_en, mod_src_en = load_model_pair(source_lang, "en")
        # Step B: EN -> Target
        tok_en_tgt, mod_en_tgt = load_model_pair("en", target_lang)

        if tok_src_en and mod_src_en and tok_en_tgt and mod_en_tgt:
            intermediate_en = run_inference(text, source_lang, "en", tok_src_en, mod_src_en)
            return run_inference(intermediate_en, "en", target_lang, tok_en_tgt, mod_en_tgt)

    raise ValueError(f"Unable to resolve translation path for {source_lang.upper()} → {target_lang.upper()}")

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