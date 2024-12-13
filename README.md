# Multimodal Human-Emotion Recognition System

This project integrates **vision-based emotion recognition** with **text sentiment analysis** to create a multimodal system for understanding human emotions. It combines predictions from both modalities to enhance the robustness and accuracy of emotion recognition, which is particularly relevant for applications in **human-robot interaction**, **social robotics**, and **mental health assessment**.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Vision Model](#vision-model)
  - [Text Sentiment Analysis](#text-sentiment-analysis)
  - [Multimodal Integration](#multimodal-integration)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Improvements](#future-improvements)

---

## Project Overview
The system combines:
1. A **vision model** trained on the FER2013 dataset to classify facial emotions such as happy, sad, angry, and neutral.
2. A **sentiment analysis model** using Hugging Face Transformers to classify text as positive, neutral, or negative.
3. A **multimodal pipeline** that merges the outputs from both models to generate a comprehensive prediction.

---

## Dataset
### Vision Dataset
- **FER2013**: A dataset of grayscale 48x48 pixel images of human faces categorized into 7 emotions:
  - `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, and `surprise`.
- Train/Test Split: 
  - **Train**: 28,709 images
  - **Test**: 7,178 images

### Text Dataset
- Example text inputs were manually curated for this project to demonstrate sentiment analysis capabilities:
  - Positive: "I am so happy today!"
  - Negative: "This is terrible!"
  - Neutral: "I feel neutral about this."

---

## Methodology

### Vision Model
- A pre-trained **ResNet18** model was fine-tuned on the FER2013 dataset.
- Key techniques used:
  - **Data Augmentation**: Horizontal flipping and random rotations to improve generalization.
  - **Dropout Regularization**: Added to prevent overfitting.
  - **Weighted Loss**: Handled class imbalances in the dataset.

### Text Sentiment Analysis
- Hugging Face's **CardiffNLP Twitter RoBERTa** model was used for text sentiment analysis.
- Outputs:
  - Positive
  - Neutral
  - Negative

### Multimodal Integration
- A pipeline was developed to combine predictions:
  - **Vision Model Output**: Emotion prediction from the facial image.
  - **Sentiment Analyzer Output**: Sentiment and confidence score from text input.
- Combined results provide a holistic understanding of emotions.

---

## Results

### Vision Model
- **Validation Accuracy**: ~59%
- **Test Accuracy**: ~58%
- Common misclassifications occurred between overlapping emotions like "angry" and "sad."

### Text Sentiment Analysis
- Sentiments were accurately classified for positive, negative, and neutral examples.

### Multimodal Predictions
- Example Results:

| Image Emotion | Text Sentiment         | Combined Output                              |
|---------------|------------------------|---------------------------------------------|
| Happy         | Positive (99%)         | Vision: Happy, Text: Positive               |
| Angry         | Negative (96%)         | Vision: Angry, Text: Negative               |
| Neutral       | Neutral (55%)          | Vision: Neutral, Text: Neutral              |

---

## How to Run
1. Clone this repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and prepare the **FER2013** dataset:
   - [FER2013 on Kaggle](https://www.kaggle.com/msambare/fer2013)
   - Place the dataset in a folder named `fer2013`.
4. Run the notebook or script:
   - Training the Vision Model: `train_vision_model.py`
   - Sentiment Analysis: Integrated directly in `multimodal_pipeline.py`
5. Evaluate the Multimodal System:
   - Run `evaluate_multimodal.py` to visualize combined predictions.

---

## Future Improvements
1. **Sequence Modeling**:
   - Extend the system to analyze sequences of emotions over time using RNNs or Transformers.
2. **Additional Modalities**:
   - Incorporate **audio** (e.g., speech sentiment) or **physiological signals**.
3. **Real-World Deployment**:
   - Develop a real-time multimodal emotion recognition system for robots or applications in healthcare.

---

## Acknowledgments
- **FER2013 Dataset**: Kaggle's Challenges in Representation Learning.
- **Hugging Face Transformers**: For pre-trained text models.
- **PyTorch**: For model development and training.

---

## License
This project is open-source and available under the MIT License.

