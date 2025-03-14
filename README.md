# Sign Language Recognition with PyTorch

This project aims to build a real-time sign language recognition system using a deep learning model built with **PyTorch**. The model is based on a pre-trained ResNet18 architecture that is fine-tuned on the **Sign Language MNIST** dataset. The system uses a webcam to capture hand gestures and predicts the corresponding American Sign Language (ASL) letter.

## Features

- **Pre-trained Model**: The project utilizes **ResNet18**, a deep convolutional neural network, pre-trained on ImageNet, and fine-tuned on the ASL dataset for recognizing sign language letters (A-Z).
- **Real-Time Recognition**: The model is capable of real-time sign language recognition using a webcam feed.
- **Letter Prediction**: The model predicts and displays the ASL letter on the video feed.

## Requirements

- Python 3.6 or higher
- PyTorch
- torchvision
- OpenCV
- NumPy

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sign-language-recognition.git
   cd sign-language-recognition
