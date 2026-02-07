# facial-verification-siamese-network
Facial verification system using a Siamese Neural Network with TensorFlow and OpenCV, learning image similarity through one-shot learning.

Facial Verification with a Siamese Network

This repository implements a Siamese Neural Network for facial verification using TensorFlow and OpenCV. Instead of performing face classification, the model learns a similarity function to determine whether two facial images belong to the same person.

Project Overview

Facial verification is a one-shot learning problem where the objective is to compare two images and determine identity similarity. This project uses a Siamese Network with shared weights to generate facial embeddings and compute similarity using a custom distance layer.

Architecture

Model Type: Siamese Network

Backbone: Convolutional Neural Network with shared weights

Input Size: 100 x 100 x 3

Distance Metric: L1 Distance

Framework: TensorFlow (Functional API)

The network processes two input images through identical CNN branches and computes the absolute difference between their embeddings to determine similarity.

Dataset and Data Collection

Negative Samples: Labeled Faces in the Wild (LFW) dataset

Anchor and Positive Samples: Captured manually using webcam integration

Data Augmentation: Implemented using TensorFlow image operations including brightness adjustment, contrast variation, horizontal flip, and JPEG quality changes

This combination improves generalization and robustness.

Training Strategy

Input Pairs:

Anchor and Positive labeled as 1

Anchor and Negative labeled as 0

Loss Function: Binary Crossentropy

Optimization: TensorFlow Functional API

GPU Optimization: Memory growth enabled to prevent out-of-memory errors

The model learns to minimize embedding distance for same-person pairs and maximize it for different-person pairs.

Preprocessing Pipeline
def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100,100))
    img = img / 255.0
    return img

All images are resized and normalized to ensure consistent input across training and inference.

Inference and Verification

The trained model compares an anchor image with a validation image and outputs a similarity score. A configurable confidence threshold determines whether the two faces belong to the same person.

Usage

Install dependencies: tensorflow, opencv-python, matplotlib

Capture anchor and positive images using the webcam script

Train the Siamese Network using paired data

Perform facial verification on new images

Key Features

One-shot facial verification

Custom L1 distance layer

Webcam-based data collection

Robust augmentation pipeline

Deployment-ready inference logic

Technologies Used

Python
TensorFlow
OpenCV
NumPy
Matplotlib

Repository Tags

Deep-Learning
Computer-Vision
Siamese-Network
Facial-Recognition
One-Shot-Learning
TensorFlow-2
