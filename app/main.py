from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.engines.argos_engine import translate_argos, ARGOS_LANGUAGES
from app.engines.indic_engine import translate_indic

app = FastAPI()
templates = Jinja2Templates(directory="templates")

INDIAN_LANGUAGES = {"hi", "ml", "ta", "te", "kn"}

def route_translation(text: str, source_lang: str, target_lang: str) -> str:
    if source_lang in INDIAN_LANGUAGES or target_lang in INDIAN_LANGUAGES:
        return translate_indic(text, source_lang, target_lang)
    else:
        return translate_argos(text, source_lang, target_lang)

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
        translated_text = route_translation(clean_text, source_lang, target_lang)
        if not translated_text:
            return '<p class="text-amber-400 font-medium text-sm">Could not translate text. Please verify language selection.</p>'
            
        return f'<p class="text-slate-100 font-medium text-sm leading-relaxed">{translated_text}</p>'
    except ValueError as ve:
        return f'<p class="text-amber-400 font-medium text-sm">{str(ve)}</p>'
    except Exception as e:
        return f'<p class="text-amber-400 font-medium text-sm">Translation error: {str(e)}</p>'