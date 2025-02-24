import os
import pickle
import numpy as np
from PIL import Image

# Url of the python version of cifar10  "https://www.cs.toronto.edu/~kriz/cifar.html" 


# Define paths
cifar_root = '' # define root path of the dataset containing batches  
output_dir = "cifar10"  # Folder where images will be saved

# CIFAR-10 class names
classes = [
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
for class_name in classes:
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

def load_batch(file_path):
    """Load a single batch file from CIFAR-10 dataset (Pickle format)."""
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')  # Load with byte encoding
    return batch

def save_cifar_images(cifar10_dir, output_dir):
    """Load CIFAR-10 batches and save images into respective class folders."""
    image_count = 0

    for i in range(1, 6):  # CIFAR-10 training set has 5 data batches
        batch_file = os.path.join(cifar10_dir, f"data_batch_{i}")
        batch = load_batch(batch_file)

        # Extract data and labels
        X_batch = batch[b'data']  # Shape: (10000, 3072)
        y_batch = batch[b'labels']  # List of 10000 labels

        # Reshape images to (32, 32, 3)
        X_batch = X_batch.reshape(-1, 3, 32, 32)  # (N, 3, 32, 32)
        X_batch = X_batch.transpose(0, 2, 3, 1)  # Convert to (N, 32, 32, 3)

        for j in range(len(X_batch)):
            img = Image.fromarray(X_batch[j])  # Convert to PIL image
            class_name = classes[y_batch[j]]  # Get class name
            
            # Define file path
            file_path = os.path.join(output_dir, class_name, f"image_{image_count}.png")
            
            # Save image
            img.save(file_path)
            image_count += 1

        print(f"✅ Processed batch {i} - Total images saved so far: {image_count}")

    print(f"✅ All images saved in {output_dir}/")

save_cifar_images(cifar_root, output_dir)
