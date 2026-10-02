from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.engines.argos_engine import translate_argos, setup_argos_models
from app.engines.indic_engine import translate_indic

# Non-blocking lifespan model loading
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Perform startup loading tasks
    try:
        setup_argos_models()
    except Exception as e:
        print(f"Argos init note: {e}")
    yield
    # Shutdown cleanup (if any)

app = FastAPI(lifespan=lifespan)

# Resolve path to Samsarikkam/templates
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

INDIAN_LANGUAGES = {"hi", "ml", "ta", "te", "kn"}

def route_translation(text: str, source_lang: str, target_lang: str) -> str:
    if source_lang in INDIAN_LANGUAGES or target_lang in INDIAN_LANGUAGES:
        return translate_indic(text, source_lang, target_lang)
    else:
        return translate_argos(text, source_lang, target_lang)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
        translated_text = route_translation(clean_text, source_lang, target_lang)
        if not translated_text:
            return '<p class="text-amber-400 font-medium text-sm">Could not translate text. Please verify language selection.</p>'
            
        return f'<p class="text-slate-100 font-medium text-sm leading-relaxed">{translated_text}</p>'
    except ValueError as ve:
        return f'<p class="text-amber-400 font-medium text-sm">{str(ve)}</p>'
    except Exception as e:
        return f'<p class="text-amber-400 font-medium text-sm">Translation error: {str(e)}</p>'