from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os
import json

app = Flask(__name__)
CORS(app)

# Global variables
model = None
class_names = None
model_metrics = None

def load_model_metrics():
    """Load model metrics from training history"""
    global model_metrics
    try:
        history_path = r"D:\ML\Plant_disease_2\backend\training_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                model_metrics = json.load(f)
                print("Model metrics loaded successfully")
                return model_metrics
        else:
            # If json doesn't exist, try numpy format
            npy_path = r"D:\ML\Plant_disease_2\backend\training_history.npy"
            if os.path.exists(npy_path):
                history = np.load(npy_path, allow_pickle=True).item()
                model_metrics = {
                    'val_accuracy': float(history['val_accuracy'][-1]),
                    'accuracy': float(history['accuracy'][-1]),
                    'val_loss': float(history['val_loss'][-1]),
                    'loss': float(history['loss'][-1])
                }
                print("Model metrics loaded successfully from numpy file")
                return model_metrics
    except Exception as e:
        print(f"Error loading model metrics: {e}")
        return None

# def create_model():
#     """Recreate the model architecture"""
#     model = keras.Sequential()
    
#     model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same",input_shape=(256,256,3)))
#     model.add(keras.layers.Conv2D(32,(3,3),activation="relu",padding="same"))
#     model.add(keras.layers.MaxPooling2D(3,3))
    
#     model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
#     model.add(keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
#     model.add(keras.layers.MaxPooling2D(3,3))
    
#     model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
#     model.add(keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
#     model.add(keras.layers.MaxPooling2D(3,3))
    
#     model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
#     model.add(keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
    
#     model.add(keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"))
#     model.add(keras.layers.Conv2D(512,(5,5),activation="relu",padding="same"))
    
#     model.add(keras.layers.Flatten())
    
#     model.add(keras.layers.Dense(1568,activation="relu"))
#     model.add(keras.layers.Dropout(0.5))
    
#     model.add(keras.layers.Dense(38,activation="softmax"))
    
#     return model

def load_model():
    global model
    try:
        model_path = r"D:\ML\Plant_disease_2\backend\plant_disease_model.keras"
        weights_path = r"D:\ML\Plant_disease_2\backend\plant_disease_model.weights.h5"
        
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            print("Model loaded successfully from .keras format")
        
        elif os.path.exists(weights_path):
            # model = create_model()
            model.load_weights(weights_path)
            opt = keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(optimizer=opt,
                         loss="sparse_categorical_crossentropy",
                         metrics=['accuracy'])
            print("Model loaded successfully from weights")
        
        else:
            raise FileNotFoundError("No model or weights file found")
        
        # Load model metrics after loading the model
        load_model_metrics()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def load_class_names():
    global class_names
    try:
        class_names_path = r"D:\ML\Plant_disease_2\backend\class_names.txt"
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                return [line.strip() for line in f]
        
        train_dir = r"D:\ML\Plant_disease_2\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
        if os.path.exists(train_dir):
            return os.listdir(train_dir)
        
        raise FileNotFoundError("Could not find class names")
    
    except Exception as e:
        print(f"Error loading class names: {e}")
        return None

def preprocess_image(image_bytes):
    """Preprocess the uploaded image for model prediction"""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((256, 256))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided', 'success': False}), 400
    
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        
        if class_names is None:
            return jsonify({
                'error': 'Class names not available',
                'raw_prediction': int(predicted_class_index),
                'confidence': float(predictions[0][predicted_class_index]),
                'success': False
            }), 500
            
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'disease': class_names[idx],
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_indices
        ]
        
        # Include model metrics in response
        model_performance = {
            'validation_accuracy': model_metrics['val_accuracy'] if model_metrics else None,
            'training_accuracy': model_metrics['accuracy'] if model_metrics else None,
            'validation_loss': model_metrics['val_loss'] if model_metrics else None,
            'training_loss': model_metrics['loss'] if model_metrics else None
        }
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'top_3_predictions': top_3_predictions,
            'model_metrics': model_performance,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    try:
        return jsonify({
            'supported_plants': list(set([name.split('___')[0] for name in class_names])) if class_names else [],
            'total_diseases': len(class_names) if class_names else 0,
            'image_requirements': {
                'width': 256,
                'height': 256,
                'format': ['jpg', 'jpeg', 'png']
            },
            'model_metrics': model_metrics
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    class_names = load_class_names()
    load_model()
    app.run(debug=True, port=5000)