from flask import Flask, request, jsonify, render_template, request, redirect, url_for
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

# ------------------------------------------------------------------------------
# Globals for model + metadata
# ------------------------------------------------------------------------------
model = None
class_names = None
model_metrics = None

# Paths for model/metrics/class names
MODEL_KERAS_PATH = r"D:\ML\Plant_disease_2\backend\plant_disease_model.keras"
MODEL_WEIGHTS_PATH = r"D:\ML\Plant_disease_2\backend\plant_disease_model.weights.h5"
HISTORY_JSON_PATH = r"D:\ML\Plant_disease_2\backend\training_history.json"
HISTORY_NPY_PATH = r"D:\ML\Plant_disease_2\backend\training_history.npy"
CLASS_NAMES_TXT_PATH = r"D:\ML\Plant_disease_2\backend\class_names.txt"
TRAIN_DIR_FALLBACK = r"D:\ML\Plant_disease_2\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"

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

        model_performance = {
            "validation_accuracy": (model_metrics or {}).get("val_accuracy"),
            "training_accuracy": (model_metrics or {}).get("accuracy"),
            "validation_loss": (model_metrics or {}).get("val_loss"),
            "training_loss": (model_metrics or {}).get("loss"),
        }

        # Store prediction in the database
        try:
            upload_record = Upload(
                filename=image_file.filename,
                predicted_class=predicted_class,
                confidence=confidence,
                top_3_predictions=pyjson.dumps(top_3_predictions),
            )
            db.session.add(upload_record)
            db.session.commit()
        except SQLAlchemyError as e:
            db.session.rollback()
            print(f"Database error: {e}")

        return jsonify(
            {
                "prediction": predicted_class,
                "confidence": confidence,
                "top_3_predictions": top_3_predictions,
                "model_metrics": model_performance,
                "success": True,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/api/uploads", methods=["GET"])
def get_uploads():
    try:
        uploads = Upload.query.order_by(Upload.timestamp.desc()).limit(10).all()
        return jsonify(
            [
                {
                    "filename": u.filename,
                    "predicted_class": u.predicted_class,
                    "confidence": float(u.confidence),
                    "top_3_predictions": pyjson.loads(u.top_3_predictions),
                    "timestamp": u.timestamp.isoformat(),
                }
                for u in uploads
            ]
        )
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

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

# ------------------------------------------------------------------------------
# Page routes
# ------------------------------------------------------------------------------

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

@app.route("/user")
def user_page():
    return render_template("user.html")

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
