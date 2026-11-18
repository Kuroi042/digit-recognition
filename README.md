# Handwritten Digit Recognition

A desktop application that recognizes a digit drawn by the user. It lets you compare a custom CNN with a MobileNetV2 transfer-learning model trained to classify handwritten digits from **0** to **9**.

## Features

- Draw a digit on an interactive canvas.
- Classify the drawing with either the custom CNN or MobileNetV2.
- Display the predicted digit and the three most likely classes.
- Normalize drawings in an MNIST-style format: crop the digit, preserve its aspect ratio, and center it on a 28 × 28 black canvas.
- Save normalized drawings as PNG files for inspection.

## Models

| Model | Description | Expected input |
| --- | --- | --- |
| Custom CNN | CNN trained from scratch for digit classification | 28 × 28 grayscale image |
| MobileNetV2 | ImageNet-pretrained MobileNetV2 with a 10-class classification head | 224 × 224 RGB image |

The GUI applies the preprocessing required by each model automatically.

## Requirements

- Python 3.12
- The dependencies listed in `req.txt`
- Saved model files in `src/mobilnet/`:
  - `my_model.keras` — custom CNN
  - `mobilennetmodel.keras` — MobileNetV2

## Run the application

From the project root:

```bash
bash run.sh
```

The script creates or reuses `.venv-py312`, installs the dependencies, and launches the GUI.

To run it manually:

```bash
python3.12 -m venv .venv-py312
source .venv-py312/bin/activate
python -m pip install -r req.txt
python gui.py
```

## Usage

1. Draw one digit using the mouse.
2. Select **Predict with CNN** or **Predict with MobileNetV2**.
3. Review the predicted digit and the top three probabilities.
4. Use **Clear** to start over, or **Save drawing** to save the normalized digit under `src/saved_digits/`.

## Project structure

```text
.
├── gui.py                  # PySide6 drawing and prediction application
├── run.sh                  # Environment setup and application launcher
├── req.txt                 # Python dependencies
└── src/
    ├── cnn.ipynb           # Custom CNN training notebook
    ├── Mobilnet.ipynb      # MobileNetV2 training notebook
    ├── mobilnet/           # Saved Keras models
    └── saved_digits/       # Drawings saved by the application
```

## Notes

- A GPU is optional. TensorFlow runs on the CPU when no compatible CUDA driver is available.
- The confidence values are model probabilities, not a guarantee that a prediction is correct.
