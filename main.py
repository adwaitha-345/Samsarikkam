from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load models for both translation directions
MODEL_ES_EN = "Helsinki-NLP/opus-mt-es-en"
MODEL_EN_ES = "Helsinki-NLP/opus-mt-en-es"

print("Loading local NMT models into memory...")

# Spanish -> English
tokenizer_es_en = MarianTokenizer.from_pretrained(MODEL_ES_EN)
model_es_en = MarianMTModel.from_pretrained(MODEL_ES_EN)

# English -> Spanish
tokenizer_en_es = MarianTokenizer.from_pretrained(MODEL_EN_ES)
model_en_es = MarianMTModel.from_pretrained(MODEL_EN_ES)

print("Both models loaded successfully!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/translate", response_class=HTMLResponse)
async def translate_text(
    request: Request, 
    text: str = Form(...),
    source_lang: str = Form("es"),
    target_lang: str = Form("en")
):
    clean_text = text.strip()
    if not clean_text:
        return '<div id="output" class="p-4 bg-slate-950/60 text-amber-400 rounded-2xl border border-amber-500/20 text-sm">Please enter text to translate.</div>'
    
    # Route to the correct model based on source language
    if source_lang == "en" and target_lang == "es":
        tokenizer = tokenizer_en_es
        model = model_en_es
    else:
        tokenizer = tokenizer_es_en
        model = model_es_en

    # Perform inference
    inputs = tokenizer(clean_text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return f'<p class="text-slate-100 font-medium text-sm leading-relaxed">{translated_text}</p>'