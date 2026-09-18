from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Cache loaded models in memory so they aren't reloaded on every request
LOADED_MODELS = {}

def get_model_and_tokenizer(source_lang: str, target_lang: str):
    """Dynamically fetch and cache MarianMT models."""
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    
    if model_name not in LOADED_MODELS:
        try:
            print(f"Loading model '{model_name}' into memory...")
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            LOADED_MODELS[model_name] = (tokenizer, model)
        except Exception:
            # Fall back to English pivot if direct translation pair is unavailable
            return None, None
            
    return LOADED_MODELS[model_name]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/translate", response_class=HTMLResponse)
async def translate_text(
    request: Request, 
    text: str = Form(""),
    source_lang: str = Form("es"),
    target_lang: str = Form("en")
):
    clean_text = text.strip()
    if not clean_text:
        return '<p class="italic text-slate-500">Please enter text to translate.</p>'
    
    if source_lang == target_lang:
        return f'<p class="text-slate-100 font-medium text-sm leading-relaxed">{clean_text}</p>'

    tokenizer, model = get_model_and_tokenizer(source_lang, target_lang)
    
    if not tokenizer or not model:
        return f'<p class="text-amber-400 font-medium text-sm">Model pair <strong>{source_lang.upper()} → {target_lang.upper()}</strong> is not available on Hugging Face.</p>'

    # Perform inference
    try:
        inputs = tokenizer(clean_text, return_tensors="pt", padding=True)
        translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        
        return f'<p class="text-slate-100 font-medium text-sm leading-relaxed">{translated_text}</p>'
    except Exception as e:
        return f'<p class="text-red-400 text-sm">Translation error: {str(e)}</p>'