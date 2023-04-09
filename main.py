from cProfile import label
from email.mime import image
from tkinter import image_names
from fastapi import FastAPI, File, UploadFile
from tensorflow import keras
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from keras.models import load_model
from keras.applications.resnet import preprocess_input
from keras.preprocessing import image
import os
import gdown

url = 'https://drive.google.com/uc?id=1aJQ6SZwt4vQwJl54EevSA5w_CdjZ61Wb'
output = 'saved_sample.h5'
gdown.download(url, output, quiet=False)

app = FastAPI()
model = load_model(
    os.path.abspath(os.getcwd()) + "/saved_sample.h5")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load the image and convert it to a numpy array
        image = Image.open(io.BytesIO(await file.read()))
        image = image.resize((224, 224))
        # image_array = np.array(image)
        image_array = tf.keras.utils.img_to_array(image)
        img_batch = np.expand_dims(image_array, axis=0)

        # Preprocess the image
        image_array = preprocess_input(img_batch.copy())

        # Load the model and make a prediction
        # with tf.keras.utils.custom_object_scope(custom_objects):

        # Remove the extra dimension from the input
        image_array = np.squeeze(image_array, axis=0)

        results = model.predict(image_array[np.newaxis, ...])

        # Load the class names
        class_names = ['Early Blight', 'Septoria',
                       'Yellow Curl', 'Healthy Leaf']

        # Add labels to the results
        results_with_labels = []
        for i, probability in enumerate(results[0]):
            label = class_names[i]
            result_with_label = {"label": label,
                                 "probability": float(probability)}
            results_with_labels.append(result_with_label)

        return results_with_labels

        # Convert the result to a list and return it
        # return {"result": result.tolist()}

    except Exception as e:
        return {"error": str(e)}
