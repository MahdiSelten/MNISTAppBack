from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel


import base64
import numpy as np
import cv2

import tensorflow as tf

app = FastAPI()


model = tf.keras.models.load_model("../MNISTModel2.keras")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://mnistappfront.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class userInput(BaseModel):
    username: str
    extension: str
    imageData: str


ALLOWED_EXTENTIONS = {"png", "jpeg", "jpg"}


def image_preprocess(image):
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threashold = 127
    _, binary = cv2.threshold(grayimage, threashold, 255, cv2.THRESH_BINARY_INV)
    binary = cv2.resize(binary, (28,28))
    binary = binary.reshape(28,28,1)
    binary = np.expand_dims(binary, axis=0)
    
    return binary

def image_preprocessPNG(image):

    if image.shape[-1] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3] / 255.0

        white_bg = np.ones_like(bgr, dtype=np.uint8) * 255

        image = (bgr * alpha[..., None] + white_bg * (1 - alpha[..., None])).astype(np.uint8)

    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(grayimage, 127, 255, cv2.THRESH_BINARY_INV)

    binary = cv2.resize(binary, (28, 28))

    binary = binary.astype("float32") / 255.0

    binary = binary.reshape(1, 28, 28, 1)

    return binary



@app.post("/predict")
async def predict(data: userInput):
    extension = data.extension.lower()
    if extension == "jpg":
        extension = "jpeg"
    if extension not in ALLOWED_EXTENTIONS:
        return {"error": f"extention{extension} is not currently processed"}
    
    try:
        image_bytes = base64.b64decode(data.imageData)
        arraypixels = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(arraypixels, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Image type is not taken into processing"}
        if extension == "jpeg":
            final_image = image_preprocess(image)
        if extension == "png":
            final_image = image_preprocessPNG(image)
        prediction = model.predict(final_image)

        return {"prediction": int(np.argmax(prediction))}
    except Exception as e:
        return {"error": str(e)}



    












