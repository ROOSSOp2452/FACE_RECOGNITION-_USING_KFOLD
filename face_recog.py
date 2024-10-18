import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix, top_k_accuracy_score

# Dataset path and image size
train_dir = r'D:\face_recog\dataset'  # Update with your dataset path
img_size = (160, 160)  # Input size for the model

# Load the training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=32,
    label_mode='categorical',
    validation_split=0.2,
    subset="training",
    seed=123
)

# Load the validation dataset
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=32,
    label_mode='categorical',
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Visualize some training images
class_names = train_dataset.class_names
print(f'Class names: {class_names}')

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i].numpy().argmax()])
        plt.axis("off")
plt.show()

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

augmented_train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Check class distribution in the training dataset
class_indices = np.concatenate([y.numpy() for _, y in train_dataset])
class_counts = Counter(np.argmax(class_indices, axis=1))
print(f"Class distribution in training data: {class_counts}")

# Build the model
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    augmented_train_dataset,
    validation_data=validation_dataset,
    epochs=100
)

# Evaluate on validation dataset
loss, accuracy = model.evaluate(validation_dataset)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Make predictions on the validation dataset
predictions = model.predict(validation_dataset)
predicted_classes = np.argmax(predictions, axis=1)

# Get true labels
true_labels = []
for images, labels in validation_dataset:
    true_labels.extend(np.argmax(labels.numpy(), axis=-1))

true_labels = np.array(true_labels)

# Calculate accuracy
accuracy = metrics.accuracy_score(true_labels, predicted_classes)
print(f'Accuracy: {accuracy:.2f}')

# Print unique classes and their counts for debugging
unique_true_labels, counts_true = np.unique(true_labels, return_counts=True)
unique_predicted_classes, counts_pred = np.unique(predicted_classes, return_counts=True)

print(f"Unique true labels (count: {len(unique_true_labels)}): {unique_true_labels}")
print(f"Unique predicted classes (count: {len(unique_predicted_classes)}): {unique_predicted_classes}")

# Calculate top-K accuracy
top_k_accuracy = top_k_accuracy_score(true_labels, predictions, k=5, labels=np.arange(len(class_names)))
print(f'Top-5 Accuracy: {top_k_accuracy:.2f}')

# Calculate confusion matrix
confusion_matrix_result = confusion_matrix(true_labels, predicted_classes)
print(confusion_matrix_result)

# Class distribution
unique, counts = np.unique(true_labels, return_counts=True)
print(f"Class distribution in validation data: {dict(zip(unique, counts))}")

# Load your trained model for inference
model = models.load_model('my_model.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess the image for prediction
def preprocess_frame(frame):
    # Convert the frame to RGB (OpenCV captures in BGR format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize the frame to the model's input size
    resized_frame = cv2.resize(rgb_frame, img_size)
    
    # Normalize the image
    normalized_frame = resized_frame / 255.0
    
    # Expand dimensions to match model's input shape (1, 160, 160, 3)
    expanded_frame = np.expand_dims(normalized_frame, axis=0)
    
    return expanded_frame

# Function to predict the class of the frame
def predict_class(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Predict using the model
    predictions = model.predict(preprocessed_frame)
    
    # Get the predicted class index
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    
    # Get the corresponding class name
    predicted_class_name = class_names[predicted_class_idx]
    
    return predicted_class_name, predicted_class_idx, predictions

# Start the webcam feed using OpenCV
cap = cv2.VideoCapture(0)  # 0 is the default camera ID

frame_limit = 100  # Define the maximum number of frames to process
frame_count = 0    # Initialize the frame counter

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face for prediction
        face_crop = frame[y:y + h, x:x + w]
        
        # Predict the class for the detected face
        predicted_label, predicted_index, predictions = predict_class(face_crop)

        # Display the predicted class name on the frame
        cv2.putText(frame, f'Predicted: {predicted_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame with the prediction label
    cv2.imshow('Face Recognition', frame)

    # Increment the frame counter
    frame_count += 1

    # Check if the frame limit has been reached
    if frame_count >= frame_limit:
        print("Frame limit reached. Turning off the camera.")
        break

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
