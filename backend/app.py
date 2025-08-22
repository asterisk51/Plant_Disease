from flask import Flask, request, jsonify, render_template, request, redirect, url_for, g
import json
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import json as pyjson
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import google.generativeai as genai
import sqlite3

app = Flask(__name__)

#------------------------------------------------------------------------------
#Multilingual
#------------------------------------------------------------------------------





#------------------------------------------------------------------------------
# Chat bot
#------------------------------------------------------------------------------

load_dotenv()


CORS(app)

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
SYSTEM_PROMPT = """
You are a helpful website assistant.
Answer user questions clearly and concisely.

Do not include navigation commands like NAVIGATE: — 
page navigation will be handled by the system.
"""


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------- Chatbot Logic ----------

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


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "").lower().strip()

    # 1. Direct keyword match (support, about, etc.)
    for keyword, route in NAV_KEYWORDS.items():
        if keyword in user_msg or user_msg == f"/{keyword}":
            return jsonify({
                "reply": f"The {keyword.capitalize()} page contains relevant information. Navigating...",
                "navigate": route
            })

    # 2. If no keyword found → fallback to LLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg}
    ]
    reply = call_gemini(messages)

    return jsonify({"reply": reply, "navigate": None})



# ------------------------------------------------------------------------------
# App setup (single instance)
# ------------------------------------------------------------------------------
BASE_DIR = r"D:\Plant_Disease\backend"  # adjust if needed
DB_PATH = os.path.join(BASE_DIR, "plant_disease.db")

# Make sure base dir exists so SQLite can create the file
os.makedirs(BASE_DIR, exist_ok=True)

app = Flask(__name__, static_folder=r"D:\Plant_Disease\static", template_folder=r"D:\Plant_Disease\templates")
CORS(app)

# Configure SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH.replace(os.sep, '/')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ------------------------------------------------------------------------------
# Database model
# ------------------------------------------------------------------------------
class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    top_3_predictions = db.Column(db.Text, nullable=False)  # Stored as JSON string
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    message = db.Column(db.String(300), nullable=False)

# ------------------------------------------------------------------------------
# Globals for model + metadata
# ------------------------------------------------------------------------------
model = None
class_names = None
model_metrics = None

# Paths for model/metrics/class names
MODEL_KERAS_PATH = r"D:\Plant_disease\backend\plant_disease_model.keras"
MODEL_WEIGHTS_PATH = r"D:\Plant_disease\backend\plant_disease_model.weights.h5"
# HISTORY_JSON_PATH = r"D:\ML\Plant_disease_2\backend\training_history.json"
HISTORY_NPY_PATH = r"D:\Plant_disease\backend\training_history.npy"
CLASS_NAMES_TXT_PATH = r"D:\Plant_disease\backend\class_names.txt"
TRAIN_DIR_FALLBACK = r"D:\ML\Plant_disease(not to be presented)\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def load_model_metrics():
    """Load training/validation metrics from JSON or NPY."""
    global model_metrics
    try:
        if os.path.exists(HISTORY_JSON_PATH):
            with open(HISTORY_JSON_PATH, "r") as f:
                model_metrics = pyjson.load(f)
                print("Model metrics loaded successfully")
                return model_metrics
        elif os.path.exists(HISTORY_NPY_PATH):
            history = np.load(HISTORY_NPY_PATH, allow_pickle=True).item()
            model_metrics = {
                "val_accuracy": float(history["val_accuracy"][-1]),
                "accuracy": float(history["accuracy"][-1]),
                "val_loss": float(history["val_loss"][-1]),
                "loss": float(history["loss"][-1]),
            }
            print("Model metrics loaded from numpy file")
            return model_metrics
        else:
            print("No metrics file found")
            model_metrics = None
            return None
    except Exception as e:
        print(f"Error loading model metrics: {e}")
        model_metrics = None
        return None

def load_model():
    """Load Keras model from .keras; if only weights exist, require create_model()."""
    global model
    try:
        if os.path.exists(MODEL_KERAS_PATH):
            model = keras.models.load_model(MODEL_KERAS_PATH)
            print("Model loaded from .keras file")
        elif os.path.exists(MODEL_WEIGHTS_PATH):
            # If you actually have a create_model() factory, import/use it here.
            raise RuntimeError(
                "Weights file found but no model definition present. "
                "Define and call create_model() to load weights."
            )
        else:
            raise FileNotFoundError("Model file not found")

        load_model_metrics()
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        # Re-raise so startup fails loudly, or comment the next line to allow server start without model
        # raise

def load_class_names():
    """Load class names from text file or fallback to directory listing."""
    global class_names
    try:
        if os.path.exists(CLASS_NAMES_TXT_PATH):
            with open(CLASS_NAMES_TXT_PATH, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(class_names)} class names from txt")
                return class_names

        if os.path.exists(TRAIN_DIR_FALLBACK):
            class_names = sorted(
                [name for name in os.listdir(TRAIN_DIR_FALLBACK) if os.path.isdir(os.path.join(TRAIN_DIR_FALLBACK, name))]
            )
            print(f"Loaded {len(class_names)} class names from train dir")
            return class_names

        raise FileNotFoundError("Class names file or fallback directory not found")
    except Exception as e:
        print(f"Error loading class names: {e}")
        class_names = None
        return None

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((256, 256))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

# ------------------------------------------------------------------------------
# API routes
# ------------------------------------------------------------------------------
def get_last_uploads(limit=10):
    conn = sqlite3.connect("plant_disease.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT image_path, disease, confidence, timestamp
        FROM uploads
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "image_url": f"/static/uploads/{row[0]}",
            "disease": row[1],
            "confidence": f"{row[2]:.2f}%",
            "date": row[3].split(" ")[0],   # e.g. 2025-08-23
            "time": row[3].split(" ")[1]    # e.g. 10:32
        }
        for row in rows
    ]

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided", "success": False}), 400

    if model is None:
        return jsonify({"error": "Model not loaded on server", "success": False}), 500

    try:
        image_file = request.files["image"]
        image_bytes = image_file.read()
        processed_image = preprocess_image(image_bytes)

        predictions = model.predict(processed_image)
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])

        if not class_names or predicted_class_index >= len(class_names):
            return jsonify(
                {
                    "error": "Class names not available or index out of range",
                    "raw_prediction": predicted_class_index,
                    "confidence": confidence,
                    "success": False,
                }
            ), 500

        predicted_class = class_names[predicted_class_index]

        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {"disease": class_names[int(idx)], "confidence": float(predictions[0][int(idx)])}
            for idx in top_3_indices
        ]

        # -------------------------------
        # Save uploaded image to static/uploads
        # -------------------------------
        UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        save_path = os.path.join(UPLOAD_FOLDER, image_file.filename)

        # Reset file pointer and save
        image_file.stream.seek(0)
        image_file.save(save_path)

        # -------------------------------
        # Store prediction in the database
        # -------------------------------
        upload_record = Upload(
            filename=image_file.filename,
            predicted_class=predicted_class,
            confidence=confidence,
            top_3_predictions=pyjson.dumps(top_3_predictions),
        )
        db.session.add(upload_record)
        db.session.commit()

        # -------------------------------
        # Response
        # -------------------------------
        model_performance = {
            "validation_accuracy": (model_metrics or {}).get("val_accuracy"),
            "training_accuracy": (model_metrics or {}).get("accuracy"),
            "validation_loss": (model_metrics or {}).get("val_loss"),
            "training_loss": (model_metrics or {}).get("loss"),
        }

        return jsonify(
            {
                "prediction": predicted_class,
                "confidence": confidence,
                "model_metrics": model_performance,
                "success": True,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500



@app.route("/workspace")
def workspace_page():
    try:
        # Fetch latest 20 predictions
        uploads = Upload.query.order_by(Upload.timestamp.desc()).limit(20).all()

        # Group by date
        crops_by_date = {}
        for u in uploads:
            date_str = u.timestamp.strftime("%d %B %Y")  # e.g. "23 August 2025"
            if date_str not in crops_by_date:
                crops_by_date[date_str] = []
            crops_by_date[date_str].append({
                "name": u.predicted_class,
                "type": "Detected Plant",
                "status": f"Confidence: {u.confidence*100:.1f}%",
                "progress": "Analyzed",
                "image": f"uploads/{u.filename}"  # will be resolved as /static/uploads/...
            })

        return render_template("workspace.html", crops=crops_by_date)

    except Exception as e:
        print("Error loading workspace:", e)
        return render_template("workspace.html", crops={})

@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    try:
        plants = list(set([name.split("___")[0] for name in (class_names or [])]))
        return jsonify(
            {
                "supported_plants": plants,
                "total_diseases": len(class_names) if class_names else 0,
                "image_requirements": {"width": 256, "height": 256, "format": ["jpg", "jpeg", "png"]},
                "model_metrics": model_metrics,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500
    

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    provider = data.get("provider")
    user_msg = data.get("message", "")
    history = data.get("history", [])

    if not any(m["role"] == "system" for m in history):
        history.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    history.append({"role": "user", "content": user_msg})

    try:
        if provider == "openai":
            reply = call_openai(history, data.get("model"))
        else:
            reply = call_gemini(history, data.get("model"))
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ------------------------------------------------------------------------------
# Page routes
# ------------------------------------------------------------------------------

@app.route("/send_message", methods=["POST"])
def send_message():
    data = request.json
    if not all(k in data for k in ("username", "email", "message")):
        return jsonify({"error": "Missing fields"}), 400

    new_msg = Contact(username=data["username"], email=data["email"], message=data["message"])
    db.session.add(new_msg)
    db.session.commit()
    return jsonify({"success": True})


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]
        # Save user details to DB here
        return redirect(url_for("home"))
    return render_template("signup.html")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect_page():
    return render_template("detect.html")

@app.route("/admin")
def admin_page():
    return render_template("admin.html")

@app.route("/workspace")
def workspace():
    try:
        # Fetch latest 20 predictions from DB
        uploads = Upload.query.order_by(Upload.timestamp.desc()).limit(20).all()

        # Group by date
        crops_by_date = {}
        for u in uploads:
            date_str = u.timestamp.strftime("%d %B %Y")  # e.g. "23 August 2025"
            if date_str not in crops_by_date:
                crops_by_date[date_str] = []
            crops_by_date[date_str].append({
                "name": u.predicted_class,
                "type": "Detected Plant",
                "status": f"Confidence: {u.confidence*100:.1f}%",
                "progress": "Analyzed",
                "image": f"uploads/{u.filename}"
            })

        return render_template("workspace.html", crops=crops_by_date)

    except Exception as e:
        print("Error loading workspace:", e)
        return render_template("workspace.html", crops={})

@app.route("/user")
def user_page():
    return render_template("user.html")

@app.route("/contact")
def contact_page():
    return render_template("contact.html")

@app.route("/support")
def support_page():
    return render_template("support.html")

@app.route("/about")
def about_us():
    return render_template("about.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html", system=SYSTEM_PROMPT)


# ------------------------------------------------------------------------------
# Main entry
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load resources before starting server
    load_class_names()
    load_model()

    # Create DB tables
    with app.app_context():
        db.create_all()

    app.run(debug=True, port=5000)