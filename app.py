from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load model and class names
def load_model_and_classes():
    model = tf.keras.models.load_model('waste_classification_model.h5')
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_model_and_classes()

# Preprocess image
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict endpoint
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        image_bytes = await file.read()
        
        # Preprocess image
        img_array = preprocess_image(image_bytes)
        
        # Predict using the model
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))  # Convert float32 to float

        # Prepare response
        confidence_scores = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}  # Ensure conversion to float
        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_scores": confidence_scores
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))