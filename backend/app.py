import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import io
import json as pyjson
import numpy as np
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image

from fastapi import FastAPI, Request, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import asynccontextmanager
import asyncio, os, httpx

import tensorflow as tf
from tensorflow import keras
import google.generativeai as genai
import gdown
from backend.translations import translations
# ------------------------------------------------------------------------------
# App initialization
# ------------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "..", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "..", "templates")
DB_PATH = os.path.join(BASE_DIR, "plant_disease.db")

# Make sure base dir exists
os.makedirs(BASE_DIR, exist_ok=True)

# Static & templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ------------------------------------------------------------------------------
# Database setup
# ------------------------------------------------------------------------------
engine = create_engine(f"sqlite:///{DB_PATH.replace(os.sep, '/')}")
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Upload(Base):
    __tablename__ = "upload"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    predicted_class = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    top_3_predictions = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(200), nullable=False)


class Contact(Base):
    __tablename__ = "contact"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    email = Column(String(100), nullable=False)
    message = Column(String(300), nullable=False)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------------------
# Multilingual
# ------------------------------------------------------------------------------
languages = {
    "en": "English",
    "as": "Assamese",
    "be": "Bengali",
    "do": "Dogri",
    "gu": "Gujarati",
    "hi": "Hindi",
    "ka": "Kannad",
    "kas": "Kashmiri",
    "kok": "konkani",
    "mai": "maithili",
    "mal": "Malayalam",
    "man": "Manipuri",
    "mar": "Marathi",
    "ne": "Nepali",
    "od": "Odia",
    "pu": "Punjabi",
    "sin": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}

#keep alive
@asynccontextmanager
async def lifespan(app: FastAPI):
    # load ML model & class names ONCE when server starts
    load_class_names()
    load_model()

    async def ping_forever():
        while True:
            try:
                url = os.environ.get("https://krishil.onrender.com")
                if url:
                    async with httpx.AsyncClient(timeout=10) as client:
                        await client.get(url)
            except Exception:
                pass
            await asyncio.sleep(600)

    task = asyncio.create_task(ping_forever())
    yield
    task.cancel()



@app.get("/language")
def get_languages():
    return languages


@app.post("/translate_page")
async def translate_page(data: dict):
    texts = data.get("texts", [])
    target_lang = data.get("target_lang", "en")

    results = []
    for t in texts:
        translated = translations.get(target_lang, {}).get(t, t)
        results.append(translated)

    return {"translations": results}


# ------------------------------------------------------------------------------
# Chat bot
# ------------------------------------------------------------------------------
load_dotenv()
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
SYSTEM_PROMPT = """
You are a helpful website assistant.
Answer user questions clearly and concisely.
Do not include navigation commands like NAVIGATE: —
page navigation will be handled by the system.
"""

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

NAV_KEYWORDS = {
    "support": "/support",
    "about": "/about_us",
    "home": "/",
    "detect": "/detect",
    "workspace": "/workspace",
    "services": "/services",
    "contact": "/contact",
}


def call_gemini(messages, model=None):
    compiled = []
    for m in messages:
        prefix = (
            "System: " if m["role"] == "system"
            else "User: " if m["role"] == "user"
            else "Assistant: "
        )
        compiled.append(f"{prefix}{m['content']}")
    compiled.append("Assistant:")

    model_obj = genai.GenerativeModel(model or DEFAULT_GEMINI_MODEL)
    resp = model_obj.generate_content("\n\n".join(compiled))
    return getattr(resp, "text", "")


@app.post("/chat")
async def chat(data: dict):
    provider = data.get("provider", "gemini")
    user_msg = data.get("message", "").lower().strip()
    history = data.get("history", [])

    if not any(m["role"] == "system" for m in history):
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    for keyword, route in NAV_KEYWORDS.items():
        if keyword in user_msg or user_msg == f"/{keyword}":
            return {
                "reply": f"The {keyword.capitalize()} page contains relevant information. Navigating...",
                "navigate": route,
            }

    history.append({"role": "user", "content": user_msg})

    if provider == "openai":
        reply = "OpenAI provider not configured yet."
    else:
        reply = call_gemini(history, data.get("model"))

    return {"reply": reply, "navigate": None}


# ------------------------------------------------------------------------------
# Model loading & helpers
# ------------------------------------------------------------------------------
model = None
class_names = None
model_metrics = None

MODEL_KERAS_PATH = r"D:\Plant_Disease\backend\plant_disease_model.keras"
MODEL_WEIGHTS_PATH = r"D:\Plant_Disease\backend\plant_disease_model.weights.h5"
HISTORY_NPY_PATH = r"D:\Plant_Disease\backend\training_history.npy"
CLASS_NAMES_TXT_PATH = r"D:\Plant_Disease\backend\class_names.txt"
TRAIN_DIR_FALLBACK = r"D:\ML\Plant_disease(not to be presented)\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"


def load_model_metrics():
    global model_metrics
    try:
        if os.path.exists(HISTORY_NPY_PATH):
            history = np.load(HISTORY_NPY_PATH, allow_pickle=True).item()
            model_metrics = {
                "val_accuracy": float(history["val_accuracy"][-1]),
                "accuracy": float(history["accuracy"][-1]),
                "val_loss": float(history["val_loss"][-1]),
                "loss": float(history["loss"][-1]),
            }
            return model_metrics
        return None
    except Exception:
        return None


def load_model():
    global model
    if os.path.exists(MODEL_KERAS_PATH):
        model = keras.models.load_model(MODEL_KERAS_PATH)
    else:
        model = None
    load_model_metrics()


def load_class_names():
    global class_names
    if os.path.exists(CLASS_NAMES_TXT_PATH):
        with open(CLASS_NAMES_TXT_PATH) as f:
            class_names = [line.strip() for line in f if line.strip()]
    elif os.path.exists(TRAIN_DIR_FALLBACK):
        class_names = sorted(
            [n for n in os.listdir(TRAIN_DIR_FALLBACK) if os.path.isdir(os.path.join(TRAIN_DIR_FALLBACK, n))]
        )
    else:
        class_names = []


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0) / 255.0
    return img_array


# ------------------------------------------------------------------------------
# Prediction routes
# ------------------------------------------------------------------------------
@app.post("/api/predict")
async def predict(image: UploadFile = File(...), db: Session = Depends(get_db)):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    image_bytes = await image.read()
    processed = preprocess_image(image_bytes)

    preds = model.predict(processed)
    predicted_class_index = int(np.argmax(preds[0]))
    confidence = float(preds[0][predicted_class_index])

    if not class_names:
        raise HTTPException(status_code=500, detail="Class names not loaded")

    predicted_class = class_names[predicted_class_index]
    top_3_indices = np.argsort(preds[0])[-3:][::-1]
    top_3_predictions = [
        {"disease": class_names[int(i)], "confidence": float(preds[0][int(i)])}
        for i in top_3_indices
    ]

    upload_folder = os.path.join(r"D:\Plant_Disease\static", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    save_path = os.path.join(upload_folder, image.filename)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    record = Upload(
        filename=image.filename,
        predicted_class=predicted_class,
        confidence=confidence,
        top_3_predictions=pyjson.dumps(top_3_predictions),
    )
    db.add(record)
    db.commit()

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "model_metrics": model_metrics,
        "success": True,
    }


@app.get("/api/model-info")
def get_model_info():
    plants = list(set([n.split("___")[0] for n in (class_names or [])]))
    return {
        "supported_plants": plants,
        "total_diseases": len(class_names or []),
        "image_requirements": {"width": 256, "height": 256, "format": ["jpg", "jpeg", "png"]},
        "model_metrics": model_metrics,
    }


# ------------------------------------------------------------------------------
# Contact & signup
# ------------------------------------------------------------------------------
@app.post("/send_message")
async def send_message(data: dict, db: Session = Depends(get_db)):
    if not all(k in data for k in ("username", "email", "message")):
        raise HTTPException(status_code=400, detail="Missing fields")

    new_msg = Contact(username=data["username"], email=data["email"], message=data["message"])
    db.add(new_msg)
    db.commit()
    return {"success": True}


@app.post("/signup")
async def signup(name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    return RedirectResponse("/", status_code=303)


# ------------------------------------------------------------------------------
# Page routes (Jinja2)
# ------------------------------------------------------------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/detect")
def detect_page(request: Request):
    return templates.TemplateResponse("detect.html", {"request": request})


@app.get("/admin")
def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


@app.get("/workspace")
def workspace(request: Request, db: Session = Depends(get_db)):
    uploads = db.query(Upload).order_by(Upload.timestamp.desc()).limit(20).all()
    crops_by_date = {}
    for u in uploads:
        date_str = u.timestamp.strftime("%d %B %Y")
        crops_by_date.setdefault(date_str, []).append(
            {
                "name": u.predicted_class,
                "type": "Detected Plant",
                "status": f"Confidence: {u.confidence*100:.1f}%",
                "progress": "Analyzed",
                "image": f"uploads/{u.filename}",
            }
        )
    return templates.TemplateResponse("workspace.html", {"request": request, "crops": crops_by_date})


@app.get("/user")
def user_page(request: Request):
    return templates.TemplateResponse("user.html", {"request": request})


@app.get("/contact")
def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


@app.get("/support")
def support_page(request: Request):
    return templates.TemplateResponse("support.html", {"request": request})


@app.get("/about")
def about_us(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/chatbot")
def chatbot(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request, "system": SYSTEM_PROMPT})


@app.get("/login")
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


pages = {
    "home": "Welcome to Krishil homepage",
    "about": "About Us page content",
    "support": "Support page content",
    "detect": "Disease Detection and Treatment page",
    "workspace": "Plant Health Workspace page",
}
page_to_endpoint = {
    "detect": "detect_page",
    "workspace": "workspace",
    "about_us": "about_us",
    "support_page": "support_page",
}


@app.get("/search")
def search(request: Request, q: str = ""):
    query = q.lower()
    results = []
    for page, content in pages.items():
        if query in page.lower() or query in content.lower():
            results.append((page, content))
    return templates.TemplateResponse(
        "search_results.html", {"request": request, "query": query, "results": results, "page_to_endpoint": page_to_endpoint}
    )



