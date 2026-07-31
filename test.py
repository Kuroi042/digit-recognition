import tensorflow as tf
from tensorflow.keras import layers, models
import visualkeras
import kagglehub

# 1. Build a basic CNN model 
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)), # <-- This MUST be here
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# 2. Automatically generate the block architecture diagram
visualkeras.layered_view(model, to_file='cnn_structure.png').show()

# 3. Download the EMNIST dataset
path = kagglehub.dataset_download("crawford/emnist")
print("Path to dataset files:", path)