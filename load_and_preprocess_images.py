# Import required libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Define directories
train_dir = 'D:/leaf_disease_dataset/train'
validation_dir = 'D:/leaf_disease_dataset/validation'

# Image dimensions and batch size
image_size = (128, 128)  # Resize to 128x128 or (224, 224) if needed
batch_size = 32

# Data Augmentation for Training Set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,           # Normalize pixel values between 0 and 1
    rotation_range=40,         # Random rotations
    width_shift_range=0.2,     # Random width shifts
    height_shift_range=0.2,    # Random height shifts
    shear_range=0.2,           # Random shear
    zoom_range=0.2,            # Random zoom
    horizontal_flip=True,      # Random horizontal flip
    fill_mode='nearest'        # How to fill in newly created pixels
)

# Data Preprocessing for Validation Set
validation_datagen = ImageDataGenerator(rescale=1.0/255)

# Create Data Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,      # Resize images to 128x128
    batch_size=batch_size,
    class_mode='categorical'     # For multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=image_size,      # Resize images to 128x128
    batch_size=batch_size,
    class_mode='categorical'     # For multi-class classification
)

# Print class indices
print(f"Classes found: {train_generator.class_indices}")
print(f"Total training images: {train_generator.samples}")
print(f"Total validation images: {validation_generator.samples}")

# Build the CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # First Convolutional Layer
    layers.MaxPooling2D(2, 2),  # Max Pooling Layer
    layers.Conv2D(64, (3, 3), activation='relu'),  # Second Convolutional Layer
    layers.MaxPooling2D(2, 2),  # Max Pooling Layer
    layers.Conv2D(128, (3, 3), activation='relu'),  # Third Convolutional Layer
    layers.MaxPooling2D(2, 2),  # Max Pooling Layer
    layers.Flatten(),  # Flatten layer to prepare for Dense layer
    layers.Dense(128, activation='relu'),  # Fully Connected layer
    layers.Dense(3, activation='softmax')  # Output layer (3 classes)
])

# Compile the Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # For multi-class classification
    metrics=['accuracy']
)

# Train the Model
history = model.fit(
    train_generator,
    epochs=10,  # Change epochs as needed
    validation_data=validation_generator
)

# Save the model
model.save('leaf_disease_detection_model.h5')
