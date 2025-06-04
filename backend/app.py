from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Global variable to store the loaded model
model = None

def load_model():
    global model
    try:
        # Load the model from the saved directory
        model_path = os.path.join(os.path.dirname(__file__), 'model')
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        # Tambahan untuk cek jenis model
        print(f"Model loaded: {type(model)}")
        print("Model summary:")
        model.summary()
        
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False
    
def preprocess_image(image_bytes):
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize image to match model input size (assuming 224x224)
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    # Check if model is loaded
    if model is None:
        if not load_model():
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded'
            }), 500
    
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No image provided'
            }), 400
            
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Preprocess image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get class with highest probability
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Map class index to label (adjust these based on your model's classes)
        class_labels = ['Early Blight', 'Late Blight', 'Healthy']
        prediction = class_labels[predicted_class]
        
        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'confidence': confidence
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    # Load model when starting the server
    load_model()
    app.run(debug=True) 