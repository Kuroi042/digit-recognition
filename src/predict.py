import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path

def predict(path):
    dirr  = Path(path)
    if not dirr.exists():
        print("empty")
        return
    png_files = list(dirr.glob("*.png"))
    if not png_files:
        print("Empty or no PNG files found")
        return None
    img = max(png_files, key=lambda x: x.stat().st_mtime)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    print(type(img))
    print(img.shape)
    print(img.dtype)
    print(img.min(), img.max())
    model =  tf.keras.models.load_model("my_model.keras")
    prediction = model.predict(img, verbose=0)
    print(prediction)
    print("Digit:", np.argmax(prediction))
    print("Confidence:", np.max(prediction))



def main():
    path="saved_digits/"
    predict(path)

if __name__ == "__main__":
    main()