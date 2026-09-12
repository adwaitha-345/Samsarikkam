from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import argostranslate.package
import argostranslate.translate
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

FROM_CODE = "es"  # Spanish
TO_CODE = "en"    # English

def setup_offline_model():
    """Download and install the translation package locally once."""
    try:
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        
        package_to_install = next(
            filter(lambda x: x.from_code == FROM_CODE and x.to_code == TO_CODE, available_packages)
        )
        download_path = package_to_install.download()
        argostranslate.package.install_from_path(download_path)
        print("Model package installed successfully!")
    except Exception as e:
        print(f"Package initialization info: {e}")

setup_offline_model()

def clean_repetition(text: str) -> str:
    """Detects and strips stuck infinite token loops like 'mainstream'."""
    words = text.split()
    if len(words) > 3 and len(set(words)) == 1:
        return words[0]
    return text

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/translate", response_class=HTMLResponse)
async def translate_text(request: Request, text: str = Form(...)):
    clean_text = text.strip()
    
    if not clean_text:
        return '<div id="output" class="p-4 bg-amber-50 border border-amber-200 rounded-md text-amber-700">Please enter text to translate.</div>'
    
    try:
        installed_languages = argostranslate.translate.get_installed_languages()
        from_lang = next(filter(lambda x: x.code == FROM_CODE, installed_languages))
        to_lang = next(filter(lambda x: x.code == TO_CODE, installed_languages))
        translation = from_lang.get_translation(to_lang)
        
        # Add temporary period for single words without ending punctuation to prevent model loop
        input_to_model = clean_text
        has_punctuation = clean_text[-1] in ".!?"
        if not has_punctuation:
            input_to_model += "."

        translated = translation.translate(input_to_model)
        
        # Remove artificial ending period if the original input didn't have one
        if not has_punctuation and translated.endswith("."):
            translated = translated[:-1]
            
        # Clean up any residual token repeats
        final_output = clean_repetition(translated)

        return f'<div id="output" class="p-4 bg-gray-100 rounded-md border text-gray-800 font-medium">{final_output}</div>'
    
    except Exception as e:
        return f'<div id="output" class="p-4 bg-red-50 border border-red-200 rounded-md text-red-700">Error: {str(e)}</div>'