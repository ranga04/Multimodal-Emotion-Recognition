# -*- coding: utf-8 -*-
"""MER.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19Hzuz8qwhCCBzGAeFIif2IcX-OBt6uUO
"""

!pip install numpy pandas opencv-python matplotlib torch torchvision transformers scikit-learn

import numpy as np
import pandas as pd
import cv2
import torch
import torchvision
from transformers import pipeline

print("Libraries installed and working successfully!")

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d msambare/fer2013

!unzip fer2013.zip -d fer2013

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally
    transforms.RandomRotation(10),  # Rotate images by +/- 10 degrees
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Path to the dataset
train_path = "fer2013/train"
test_path = "fer2013/test"

# Load the train and test datasets
train_data = datasets.ImageFolder(root=train_path, transform=transform)
test_data = datasets.ImageFolder(root=test_path, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Map class indices to emotions
emotions_map = {idx: emotion for emotion, idx in train_data.class_to_idx.items()}
print("Class to Emotion Mapping:", emotions_map)

# Visualize some sample images
def visualize_data(data_loader, emotions_map):
    images, labels = next(iter(data_loader))
    plt.figure(figsize=(10, 5))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].squeeze(0), cmap="gray")
        plt.title(emotions_map[labels[i].item()])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_data(train_loader, emotions_map)

# Print dataset sizes
print(f"Training set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Modify the first convolutional layer to accept 1-channel input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify the final fully connected layer for FER2013
num_classes = 7  # FER2013 has 7 emotion classes
# Replace the final layer with a fully connected + dropout
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # Dropout with 50% probability
    nn.Linear(512, num_classes)
)

# Move the model to the GPU/CPU
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.optim.lr_scheduler import StepLR

# Define the learning rate scheduler
scheduler = StepLR(optimizer, step_size=2, gamma=0.5)  # Halve LR every 2 epochs

def train_model_with_scheduler(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Training Loss: {train_loss/len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Step the scheduler
        scheduler.step()

train_model_with_scheduler(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=10)

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Evaluate the model on the test set
def evaluate_model(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(emotions_map.values()))
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.title("Confusion Matrix")
    plt.show()

# Call the evaluation function
evaluate_model(model, test_loader, criterion)

def visualize_predictions(model, test_loader, emotions_map):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 6))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].cpu().squeeze(0), cmap="gray")
        plt.title(f"True: {emotions_map[labels[i].item()]}\nPred: {emotions_map[preds[i].item()]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Call the visualization function
visualize_predictions(model, test_loader, emotions_map)

!pip install transformers

from transformers import pipeline

# Load a sentiment analysis model that includes neutral sentiment
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=0)

# Test Sentiment Analysis
texts = [
    "I am so happy today!",  # Positive sentiment
    "This is the worst day of my life.",  # Negative sentiment
    "I feel neutral about this."  # Neutral sentiment
]

# Analyze sentiments
sentiments = sentiment_analyzer(texts)
# Mapping labels to sentiments
sentiment_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

# Display the results with mapped labels
for text, sentiment in zip(texts, sentiments):
    sentiment_label = sentiment_map[sentiment['label']]
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment_label}, Confidence: {sentiment['score']:.2f}")

def multimodal_predict(image, text, vision_model, sentiment_analyzer, emotions_map, sentiment_map):
    # Vision prediction
    vision_model.eval()
    with torch.no_grad():
        image = image.to(device)
        image = image.unsqueeze(0)  # Add batch dimension (1, 1, 48, 48)
        output = vision_model(image)  # Pass image through the model
        _, vision_pred = torch.max(output, 1)
        vision_emotion = emotions_map[vision_pred.item()]

    # Text sentiment prediction
    sentiment = sentiment_analyzer(text)[0]
    sentiment_label = sentiment_map[sentiment['label']]
    sentiment_confidence = sentiment['score']

    # Combine predictions
    combined_result = {
        "Vision Prediction": vision_emotion,
        "Text Sentiment": sentiment_label,
        "Sentiment Confidence": sentiment_confidence
    }

    return combined_result

# Example test input
image, label = next(iter(test_loader))
sample_image = image[0]  # Select the first image in the batch
sample_text = "I am very happy with this result!"

# Define mappings
emotions_map = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
sentiment_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}

# Get multimodal predictions
result = multimodal_predict(sample_image, sample_text, model, sentiment_analyzer, emotions_map, sentiment_map)

# Display the combined result
print("Multimodal Prediction:")
print(result)

def evaluate_multimodal(test_loader, vision_model, sentiment_analyzer, emotions_map, sentiment_map):
    vision_model.eval()  # Set vision model to evaluation mode
    images, labels = next(iter(test_loader))  # Get a batch of images and labels
    text_inputs = [
        "I am so happy!",  # Positive
        "This is terrible!",  # Negative
        "I feel neutral about this."  # Neutral
    ]

    # Repeat text inputs to match the batch size
    text_inputs = text_inputs * (len(images) // len(text_inputs)) + text_inputs[:len(images) % len(text_inputs)]

    # Process predictions
    predictions = []
    for i in range(len(images)):
        image = images[i]
        text = text_inputs[i]
        prediction = multimodal_predict(image, text, vision_model, sentiment_analyzer, emotions_map, sentiment_map)
        predictions.append(prediction)

    # Visualize predictions
    plt.figure(figsize=(12, 8))
    for i in range(6):  # Visualize first 6 examples
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i].squeeze(0), cmap="gray")
        plt.title(f"Vision: {predictions[i]['Vision Prediction']}\n"
                  f"Text: {text_inputs[i]}\n"
                  f"Sentiment: {predictions[i]['Text Sentiment']} ({predictions[i]['Sentiment Confidence']:.2f})")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Call the evaluation function
evaluate_multimodal(test_loader, model, sentiment_analyzer, emotions_map, sentiment_map)
