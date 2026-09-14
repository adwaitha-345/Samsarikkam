from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Spanish -> English lightweight local model
MODEL_NAME = "Helsinki-NLP/opus-mt-es-en"

print("Loading local NMT model into memory...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/translate", response_class=HTMLResponse)
async def translate_text(request: Request, text: str = Form(...)):
    clean_text = text.strip()
    if not clean_text:
        return '<div id="output" class="p-4 bg-yellow-50 text-yellow-700 rounded-md border border-yellow-200">Please enter text to translate.</div>'
    
    # Local NMT Inference
    inputs = tokenizer(clean_text, return_tensors="pt", padding=True)
    translated_tokens = model.generate(**inputs)
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return f'<div id="output" class="p-4 bg-gray-100 rounded-md border text-gray-800">{translated_text}</div>'