import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Define paths to the train and validation directories
train_dir = 'D:/leaf_disease_dataset/train'  # Modify this path
validation_dir = 'D:/leaf_disease_dataset/validation'  # Modify this path

# Resize images to a consistent size
image_size = (128, 128)  # Resize images to 128x128 pixels

# Set up the ImageDataGenerators for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=40,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Random shear transformation
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill pixels that are left blank by transformations
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess the training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'  # For multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical'
)

# Build the model using Convolutional Neural Networks (CNN)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 classes (healthy, rust, powdery_mildew)
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # You can increase this depending on your dataset and hardware
    validation_data=validation_generator
)

# Save the trained model
model.save('leaf_disease_detection_model.h5')
print("Model trained and saved successfully!")

# Test the model (You can modify this part to predict on an image)
# Here we assume you have a test image to classify
img_path = r"E:\Downloads\test1.webp"  # Replace with your test image path

test_img = image.load_img(img_path, target_size=image_size)
test_img_array = image.img_to_array(test_img)
test_img_array = np.expand_dims(test_img_array, axis=0)  # Add batch dimension
test_img_array = test_img_array / 255.0  # Normalize the image

# Predict the class of the test image
predictions = model.predict(test_img_array)
class_names = list(train_generator.class_indices.keys())  # Get class names from the training set

# Find the predicted class
predicted_class = class_names[np.argmax(predictions)]
print(f"Predicted class: {predicted_class}")
