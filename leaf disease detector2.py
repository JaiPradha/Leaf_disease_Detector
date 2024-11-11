import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import numpy as np
from tensorflow.keras.preprocessing import image

# Define paths to the train and validation directories (do not change)
train_dir = 'D:/leaf_disease_dataset/train'  # Modify this path if needed
validation_dir = 'D:/leaf_disease_dataset/validation'  # Modify this path if needed

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

# Dictionary of remedies and crop maintenance tips for each disease class
remedies = {
    'healthy': {
        'remedy': """
        No disease detected. Your crops are healthy and thriving. 
        General maintenance tips:
        1. Ensure consistent watering, avoiding overwatering or underwatering.
        2. Provide adequate sunlight for at least 6-8 hours a day.
        3. Perform soil testing for pH and nutrient levels regularly.
        4. Use organic fertilizers during the growing season to encourage healthy growth.
        5. Inspect your crops periodically for pests, weeds, or diseases, even when they seem healthy.
        """,
        'crop_maintenance': """
        - For **Tomato Plants**: Prune lower leaves to avoid soil contact and improve air circulation. Mulch around the base to retain moisture.
        - For **Corn**: Ensure a minimum of 30 cm spacing between rows for proper airflow. Fertilize with a nitrogen-rich fertilizer during early growth stages.
        - For **Potatoes**: Use raised beds to avoid waterlogging and ensure proper drainage. Apply compost to enrich the soil with nutrients.
        """
    },
    'rust': {
        'remedy': """
        Rust disease detected. Remedy:
        1. **Remove infected leaves**: Rust fungi appear as orange or red spots. Remove and destroy infected leaves to prevent further spread.
        2. **Apply fungicides**: Use copper-based or sulfur-based fungicides to treat the infection.
        3. **Increase plant spacing**: Improve air circulation around plants by increasing spacing to reduce humidity.
        4. **Rotate crops**: Avoid planting the same crop in the same soil each year to reduce rust build-up.
        5. **Water carefully**: Water plants at the base to avoid wetting leaves and promoting rust growth.
        """,
        'crop_maintenance': """
        - For **Tomato Plants**: Improve airflow by spacing plants apart and pruning any dense foliage. Water plants at the soil level.
        - For **Corn**: Rotate crops annually to avoid rust spores overwintering in the soil. Apply a fungicide early in the season as a preventive measure.
        - For **Potatoes**: After harvesting, rotate the crop with non-solanaceous plants to reduce the risk of rust reinfection.
        """
    },
    'powdery_mildew': {
        'remedy': """
        Powdery mildew detected. Remedy:
        1. **Remove infected leaves**: Powdery mildew shows up as white, powdery spots. Prune infected leaves and destroy them to stop further contamination.
        2. **Apply fungicides**: Use sulfur-based fungicides or neem oil to control the spread of the mildew.
        3. **Increase spacing**: Ensure there is proper spacing between plants to allow better air circulation, reducing humidity and mildew growth.
        4. **Avoid overhead watering**: Water plants at the base to keep moisture off the leaves.
        5. **Mulch**: Apply a thick layer of mulch to reduce soil-borne spores from splashing onto the plants.
        """,
        'crop_maintenance': """
        - For **Tomato Plants**: Apply mulch around the base to help retain moisture and keep leaves dry. Use resistant varieties if possible.
        - For **Corn**: Plant corn in full sunlight and ensure that rows are spaced adequately for proper airflow.
        - For **Potatoes**: Use drip irrigation to avoid leaf wetness, and ensure adequate spacing between plants to reduce the likelihood of mildew.
        """
    }
}

# Request the user to input the path for the test image
img_path = input("Please enter the path to your test image (e.g., E:/Downloads/test1.webp): ")

# Check if the file exists
if not os.path.exists(img_path):
    print("Error: The image file does not exist. Please check the path and try again.")
else:
    # Load and preprocess the test image
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

    # Output the remedy for the predicted disease along with crop-specific maintenance
    remedy = remedies.get(predicted_class, {"remedy": "No remedy available for this disease.", "crop_maintenance": "No maintenance tips available."})
    print(f"Recommended remedy:\n{remedy['remedy']}")
    print(f"Crop-specific maintenance:\n{remedy['crop_maintenance']}")
