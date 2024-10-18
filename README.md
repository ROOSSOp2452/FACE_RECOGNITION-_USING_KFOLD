# FACE_RECOGNITION-_USING_KFOLD
FACE_RECOGNITION 
# Face Recognition System Using Deep Learning and OpenCV

This repository contains the code for a **Real-Time Face Recognition System** built using **TensorFlow**, **Keras**, and **OpenCV**. The system utilizes a deep learning model to recognize faces in real-time video feeds, integrating a pre-trained MobileNetV2 model for feature extraction and OpenCV's Haar Cascade for face detection.

## Features
- **Face Detection**: Detects faces in real-time from webcam feeds using OpenCV's Haar Cascade.
- **Deep Learning-based Face Recognition**: Leverages a transfer-learned MobileNetV2 model fine-tuned for face recognition.
- **Data Augmentation**: Improves the model's robustness and generalization with random flipping and rotation during training.
- **Real-Time Prediction**: Predicts the identity of detected faces in real-time with the option to display prediction labels on video frames.
- **Evaluation Metrics**: Accuracy, Top-K Accuracy, and Confusion Matrix to assess model performance.

## Table of Contents
- [Dataset Preparation](#dataset-preparation)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Real-Time Face Recognition](#real-time-face-recognition)
- [Evaluation](#evaluation)
- [Technologies Used](#technologies-used)
- [Challenges Faced](#challenges-faced)
- [Future Improvements](#future-improvements)

## Dataset Preparation
The dataset is structured into subdirectories, with each directory containing images of a single individual. The directory names correspond to the class labels (i.e., the names of the individuals).

To preprocess the dataset:
1. **Resize** all images to \(160 \times 160\) pixels.
2. Apply **data augmentation** (random flips and rotations) during training for improved model performance.

## Installation
To run this project, you will need Python 3.6+ and the following libraries:
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

You can install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/face-recognition-system.git
cd face-recognition-system
```

### 2. Dataset Setup
Place your dataset in a directory with the following structure:
```
dataset/
    ├── person1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    ├── person2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    └── person3/
        ├── image1.jpg
        └── image2.jpg
```

### 3. Training the Model
To train the face recognition model, run:
```bash
python train_model.py --data_path path_to_your_dataset --epochs 100
```
This will train a model using MobileNetV2 as the base model and save the trained model as `my_model.h5`.

### 4. Real-Time Face Recognition
To run the real-time face recognition system using your webcam:
```bash
python real_time_recognition.py --model_path my_model.h5
```
The system will detect faces, recognize the individuals in real-time, and display the predicted name on the video frame.

## Training
The model is trained using **MobileNetV2** as a base model with transfer learning. Data augmentation is applied to enrich the training dataset. The final classifier is a fully connected layer with softmax activation for multi-class classification.

**Training Script**: `train_model.py`

## Real-Time Face Recognition
The real-time system utilizes OpenCV's Haar Cascade to detect faces and the trained deep learning model to classify detected faces. The prediction label is displayed on the video frame in real-time.

**Real-Time Script**: `real_time_recognition.py`

## Evaluation
Evaluation metrics such as **Accuracy**, **Top-5 Accuracy**, and **Confusion Matrix** are used to assess the performance of the model on the validation set.

```bash
python evaluate_model.py --model_path my_model.h5
```

### Key Evaluation Metrics:
- **Accuracy**: Measures the overall correctness of the model.
- **Top-5 Accuracy**: Checks if the true label is within the top 5 predictions.
- **Confusion Matrix**: Visualizes the model's classification performance.

## Technologies Used
- **TensorFlow** & **Keras**: For building and training the CNN model.
- **MobileNetV2**: Pre-trained model for efficient and effective feature extraction.
- **OpenCV**: For real-time face detection using Haar Cascades.
- **NumPy & Matplotlib**: For data manipulation and visualization.
- **Scikit-learn**: For evaluation metrics and performance analysis.

## Challenges Faced
1. **Data Imbalance**: Some classes had fewer samples, which led to biased model predictions.
   - **Solution**: Applied data augmentation to underrepresented classes to balance the dataset.
   
2. **Overfitting**: The model overfitted the training set, leading to poor generalization on validation data.
   - **Solution**: Added dropout layers and reduced model complexity to mitigate overfitting.

3. **Latency in Real-Time Inference**: Achieving real-time performance while maintaining accuracy.
   - **Solution**: Optimized preprocessing and reduced the model inference time by using a smaller, efficient network like MobileNetV2.

## Future Improvements
- **Model Optimization**: Further optimize the model for real-time inference, possibly deploying it on mobile or edge devices.
- **Face Mask Detection**: Add functionality to detect whether individuals are wearing masks and adjust face recognition performance accordingly.
- **Larger Dataset**: Incorporate a larger and more diverse dataset to improve the system's robustness across different individuals and environments.

---

Feel free to edit or add more sections as you see fit. This should give a good starting point for your GitHub `README.md` file.
