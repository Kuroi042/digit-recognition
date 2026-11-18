import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ROOT = Path(__file__).resolve().parent
CNN_MODEL_PATH = ROOT / "src" / "mobilnet" / "my_model.keras"
MOBILENET_MODEL_PATH = ROOT / "src" / "mobilnet" / "mobilennetmodel.keras"
SAVE_DIR = ROOT / "src" / "saved_digits"
CANVAS_SIZE = 500
STROKE_WIDTH = 18


def normalize_digit(image: Image.Image) -> Image.Image | None:
    """Create a centered 28x28 MNIST-style digit from a canvas image."""
    image = image.convert("L")
    bbox = image.getbbox()
    if bbox is None:
        return None
    digit = image.crop(bbox)
    width, height = digit.size
    scale = 20 / max(width, height)
    digit = digit.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
    result = Image.new("L", (28, 28), 0)
    result.paste(digit, ((28 - digit.width) // 2, (28 - digit.height) // 2))
    return result


def cnn_input(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[np.newaxis, ...]


def mobilenet_input(image: Image.Image) -> tf.Tensor:
    array = tf.convert_to_tensor(np.asarray(image, dtype=np.float32)[..., np.newaxis])
    array = tf.image.grayscale_to_rgb(array)
    array = tf.image.resize(array, (224, 224))
    return tf.expand_dims(preprocess_input(array), axis=0)


def prediction_text(probabilities: np.ndarray) -> tuple[int, str]:
    probabilities = probabilities.ravel()
    top_indices = np.argsort(probabilities)[-3:][::-1]
    return int(top_indices[0]), "\n".join(f"{index}: {probabilities[index]:.1%}" for index in top_indices)


class DrawingCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(CANVAS_SIZE, CANVAS_SIZE)
        self.last_point = None
        self.clear()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.position().toPoint()
            self.draw_point(self.last_point)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.last_point is not None:
            current = event.position().toPoint()
            painter = QPainter(self.pixmap)
            pen = QPen(QColor("white"), STROKE_WIDTH)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(self.last_point, current)
            painter.end()
            self.setPixmap(self.pixmap)
            self.draw.line([self.last_point.x(), self.last_point.y(), current.x(), current.y()], fill=255, width=STROKE_WIDTH)
            self.last_point = current

    def mouseReleaseEvent(self, _event):
        self.last_point = None

    def draw_point(self, point):
        painter = QPainter(self.pixmap)
        painter.setBrush(QColor("white"))
        painter.setPen(Qt.NoPen)
        radius = STROKE_WIDTH // 2
        painter.drawEllipse(point, radius, radius)
        painter.end()
        self.setPixmap(self.pixmap)
        self.draw.ellipse([point.x() - radius, point.y() - radius, point.x() + radius, point.y() + radius], fill=255)

    def clear(self):
        self.pixmap = QPixmap(CANVAS_SIZE, CANVAS_SIZE)
        self.pixmap.fill(Qt.black)
        self.setPixmap(self.pixmap)
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)


class DigitPad(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Handwritten Digit Recognition")
        self.cnn_model = None
        self.mobilenet_model = None
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        layout = QVBoxLayout(self)
        title = QLabel("Draw one digit (0–9), then choose a model")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        self.canvas = DrawingCanvas()
        layout.addWidget(self.canvas)

        buttons = QHBoxLayout()
        cnn_button = QPushButton("Predict with CNN")
        mobile_button = QPushButton("Predict with MobileNetV2")
        clear_button = QPushButton("Clear")
        save_button = QPushButton("Save drawing")
        cnn_button.clicked.connect(self.predict_cnn)
        mobile_button.clicked.connect(self.predict_mobilenet)
        clear_button.clicked.connect(self.clear_canvas)
        save_button.clicked.connect(self.save_image)
        for button in (cnn_button, mobile_button, clear_button, save_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)

    def current_digit(self) -> Image.Image | None:
        digit = normalize_digit(self.canvas.image)
        if digit is None:
            QMessageBox.warning(self, "Nothing to predict", "Please draw a digit first.")
        return digit

    def get_cnn_model(self):
        if self.cnn_model is None:
            self.cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        return self.cnn_model

    def get_mobilenet_model(self):
        if self.mobilenet_model is None:
            self.mobilenet_model = tf.keras.models.load_model(MOBILENET_MODEL_PATH)
        return self.mobilenet_model

    def show_prediction(self, model_name: str, probabilities: np.ndarray):
        digit, top_three = prediction_text(probabilities)
        self.status.setText(f"{model_name} predicts: {digit}")
        QMessageBox.information(self, f"{model_name} prediction", f"Predicted digit: {digit}\n\nTop predictions\n{top_three}")

    def predict_cnn(self):
        digit = self.current_digit()
        if digit is None:
            return
        try:
            self.show_prediction("Custom CNN", self.get_cnn_model().predict(cnn_input(digit), verbose=0))
        except Exception as error:
            QMessageBox.critical(self, "CNN error", str(error))

    def predict_mobilenet(self):
        digit = self.current_digit()
        if digit is None:
            return
        try:
            self.show_prediction("MobileNetV2", self.get_mobilenet_model().predict(mobilenet_input(digit), verbose=0))
        except Exception as error:
            QMessageBox.critical(self, "MobileNetV2 error", str(error))

    def clear_canvas(self):
        self.canvas.clear()
        self.status.setText("")

    def save_image(self):
        digit = self.current_digit()
        if digit is None:
            return
        filepath = SAVE_DIR / f"digit_{int(time.time())}.png"
        digit.save(filepath)
        self.status.setText(f"Saved: {filepath.relative_to(ROOT)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DigitPad()
    window.show()
    sys.exit(app.exec())
