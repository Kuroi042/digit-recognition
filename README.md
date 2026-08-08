# digit-recognition

<img width="960" height="720" alt="image" src="https://miro.medium.com/v2/resize:fit:1400/1*hSbHWUSQreLeyA6Nr2KIqg.png" />

# 0–9 Digit Recognition with CNN & MobileNetV2

A handwritten digit recognition project that compares two different deep learning approaches using TensorFlow/Keras.

The application allows the user to draw a digit from **0 to 9** through a graphical interface and predicts the digit using a trained neural network.

## Models

This project contains two Keras models:

### 1. Custom CNN

The first model is a convolutional neural network built from scratch using Keras CNN layers.

The architecture includes:

- Convolutional layers
- ReLU activation
- MaxPooling layers
- Flatten
- Dense layers
- Softmax output layer

The model is trained specifically for handwritten digit classification.

### 2. MobileNetV2 Transfer Learning

The second model uses **MobileNetV2** pretrained on the **ImageNet** dataset.

Instead of training the entire network from scratch:

- MobileNetV2 is loaded with ImageNet pretrained weights.
- The original classification head is removed.
- The pretrained convolutional base is used as a feature extractor.
- A new classification head is added for the 10 digit classes.
- MNIST images are converted from grayscale to RGB.
- Images are resized to `224 × 224`, matching the MobileNetV2 input size.
- MobileNetV2 preprocessing is applied before prediction.

This approach demonstrates the use of **transfer learning** for computer vision.

---

## Dataset

The project uses the **MNIST handwritten digit dataset**.

MNIST contains grayscale images of handwritten digits from:

```text
0 → 9
