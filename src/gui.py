import sys
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input

IMAGE_SHAPE = (224, 224)
PATH = "saved_digits/"


from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
)


CANVAS_SIZE = 500
OUTPUT_SIZE = 224
STROKE_WIDTH = 18
SAVE_DIR = "saved_digits"

os.makedirs(SAVE_DIR, exist_ok=True)


def preprocess_digit(image_path):
    img = Image.open(image_path).convert("L")
    img = img.resize((OUTPUT_SIZE, OUTPUT_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, OUTPUT_SIZE, OUTPUT_SIZE, 1)
    return arr


class DrawingCanvas(QLabel):

    def __init__(self):
        super().__init__()

        self.setFixedSize(CANVAS_SIZE, CANVAS_SIZE)

        self.pixmap = QPixmap(CANVAS_SIZE, CANVAS_SIZE)
        self.pixmap.fill(Qt.black)
        self.setPixmap(self.pixmap)

        self.last_point = None

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.last_point = event.position().toPoint()

            self.draw_point(self.last_point)

    def mouseMoveEvent(self, event):

        if event.buttons() & Qt.LeftButton:

            current = event.position().toPoint()

            painter = QPainter(self.pixmap)
            pen = QPen(QColor("white"), STROKE_WIDTH)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)

            painter.drawLine(self.last_point, current)
            painter.end()

            self.setPixmap(self.pixmap)

            self.draw.line(
                [
                    self.last_point.x(),
                    self.last_point.y(),
                    current.x(),
                    current.y(),
                ],
                fill=255,
                width=STROKE_WIDTH,
            )

            self.last_point = current

    def mouseReleaseEvent(self, event):
        self.last_point = None

    def draw_point(self, point):

        painter = QPainter(self.pixmap)

        painter.setBrush(QColor("white"))
        painter.setPen(Qt.NoPen)

        r = STROKE_WIDTH // 2

        painter.drawEllipse(point, r, r)
        painter.end()

        self.setPixmap(self.pixmap)

        self.draw.ellipse(
            [
                point.x() - r,
                point.y() - r,
                point.x() + r,
                point.y() + r,
            ],
            fill=255,
        )

    def clear(self):

        self.pixmap.fill(Qt.black)
        self.setPixmap(self.pixmap)

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)


class DigitPad(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Digit Drawing Pad")

        layout = QVBoxLayout(self)

        title = QLabel("Draw a digit (0-9)")
        title.setAlignment(Qt.AlignCenter)

        layout.addWidget(title)

        self.canvas = DrawingCanvas()

        layout.addWidget(self.canvas)

        buttons = QHBoxLayout()

        predict_btn = QPushButton("Predict")
        clear_btn = QPushButton("Clear")
        save_btn = QPushButton("Save")

        predict_btn.clicked.connect(self.predict)
        clear_btn.clicked.connect(self.clear_canvas)
        save_btn.clicked.connect(self.save_image)

        buttons.addWidget(predict_btn)
        buttons.addWidget(clear_btn)
        buttons.addWidget(save_btn)

        layout.addLayout(buttons)

        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.status)

    def predict(PATH):
        path="saved_digits/"
        dirr  = Path(path)
        if not dirr.exists():
            print("empty")
            return
        png_files = list(dirr.glob("*.png"))
        if not png_files:
            print("Empty or no PNG files found")
            return None
        img = max(png_files, key=lambda x: x.stat().st_mtime)
        img = Image.open(img)
        img = img.convert("L")
        img = img.resize((28, 28))
        img = np.array(img)
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32) # (28, 28)
        img_tensor = tf.expand_dims(img_tensor, -1)              # (28, 28, 1)
        img_tensor = tf.image.grayscale_to_rgb(img_tensor)       # (28, 28, 3)
        img_tensor = tf.image.resize(img_tensor, IMAGE_SHAPE) 
        img_tensor = mobilenet_preprocess_input(img_tensor) 
        img_tensor = tf.expand_dims(img_tensor, axis=0)          # (1, 224, 224, 3) Add batch dimension
        print(img_tensor.shape)
        model =  tf.keras.models.load_model("mobilnet/mobilennetmodel.keras")

        prediction = model.predict(img_tensor, verbose=0)
        print(prediction)
        print("Digit:", np.argmax(prediction))
        print("Confidence:", np.max(prediction))
        msg = QMessageBox()
        msg.setWindowTitle("Prediction")
        msg.setText(
            f"Predicted Digit: {np.argmax(prediction)}\n"
            f"Confidence: {np.max(prediction):.2%}"
        )
        msg.setIcon(QMessageBox.Information)
        msg.exec()

    def clear_canvas(self):
        self.canvas.clear()
        self.status.setText("")

    def save_image(self):

        small = self.canvas.image.resize(
            (OUTPUT_SIZE, OUTPUT_SIZE),
            Image.LANCZOS,
        )

        filename = f"digit_{int(time.time())}.png"

        filepath = os.path.join(SAVE_DIR, filename)

        small.save(filepath)

        arr = preprocess_digit(filepath)

        print(arr.shape)

        self.status.setText(f"Saved: {filepath}")

        QMessageBox.information(
            self,
            "Saved",
            f"Digit saved as\n\n{filepath}",
        )


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = DigitPad()
    window.show()

    sys.exit(app.exec())
