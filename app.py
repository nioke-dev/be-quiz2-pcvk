from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
import os

# Fungsi untuk mengunduh model dari Google Drive
def download_model_from_drive(drive_file_id, destination_path):
    url = f"https://drive.google.com/uc?id={drive_file_id}&export=download"
    
    if not os.path.exists(destination_path):
        try:
            print("Downloading model from Google Drive...")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for HTTP requests that fail
            with open(destination_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Model downloaded to {destination_path}.")
        except Exception as e:
            raise RuntimeError(f"Failed to download the model: {str(e)}")
    else:
        print("Model already exists locally.")

# ID file Google Drive dan path penyimpanan model
drive_file_id = "1f3AA4QxKFLM3rkxgVbvPOohWwfODev7u"  # Ganti dengan ID file Anda
destination_path = "waste_classification_model.h5"

# Unduh model jika belum ada
download_model_from_drive(drive_file_id, destination_path)

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Fungsi untuk memuat model dan nama kelas
def load_model_and_classes():
    try:
        model = tf.keras.models.load_model('waste_classification_model.h5')
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return model, class_names
    except Exception as e:
        raise RuntimeError(f"Failed to load model or class names: {str(e)}")

# Muat model dan kelas
try:
    model, class_names = load_model_and_classes()
except RuntimeError as e:
    print(str(e))
    raise SystemExit(1)  # Keluar jika gagal memuat model atau kelas

# Fungsi untuk memproses gambar
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to preprocess the image: {str(e)}")

# Endpoint untuk prediksi
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Baca file gambar yang diunggah
        image_bytes = await file.read()

        # Preproses gambar
        img_array = preprocess_image(image_bytes)

        # Lakukan prediksi
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))  # Konversi float32 ke float

        # Skor kepercayaan untuk semua kelas
        confidence_scores = {
            class_names[i]: float(predictions[0][i])
            for i in range(len(class_names))
        }

        # Berikan respons
        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_scores": confidence_scores
        })
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
