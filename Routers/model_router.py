from fastapi import APIRouter, File, HTTPException, status
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from io import BytesIO

router = APIRouter(
    tags=['Models'],
)


animals10Model = tf.keras.models.load_model("Animals10")

def animals10(file):
    try:
        file_obj = BytesIO(file)
        pil_obj = Image.open(file_obj)
        cv2_image = cv2.cvtColor(np.array(pil_obj), cv2.COLOR_BGR2RGB)

        image_array = cv2.resize(cv2_image ,(224,224))/255
        class_name = animals10Model.predict(image_array.reshape(1,224,224,3)).argmax()         #reshape(1 image , (224,224,3)size)
        ansToGive = ["Butterfly", "Cat", "Chicken", "Cow", "Dog", "Elephant", "Horse", "Sheep", "Spider", "Squirrel"]
        return str(ansToGive[class_name])
    except:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@router.post("/models/10animals")
async def create_file(file: bytes = File(...) ):
    # time.sleep(2)
    return {"prediction": animals10(file)}
