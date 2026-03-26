# Stress Detection using BERT (NLP Project)

## 📌 Overview
This project focuses on detecting stress levels from textual data using Natural Language Processing (NLP). It leverages a BERT-based transformer model to classify whether a given text indicates stress or not.

## 🚀 Features
- Uses state-of-the-art BERT model for contextual understanding
- Trained on real-world Reddit data (Dreaddit dataset)
- High accuracy (~89%)
- Handles real-world noisy textual inputs
- Demonstrates fine-tuning of transformer models

## 🧠 Model Details
- Model: BERT (Bidirectional Encoder Representations from Transformers)
- Task: Binary Text Classification (Stress vs No Stress)
- Dataset: Dreaddit (Reddit posts labeled for stress)
- Accuracy: ~89%

## ⚙️ Tech Stack
- Python
- PyTorch / Transformers (Hugging Face)
- NumPy, Pandas
- Scikit-learn

## 🔄 Workflow
1. Data Collection (Dreaddit dataset)
2. Text Preprocessing (cleaning, tokenization)
3. BERT Tokenization
4. Model Fine-Tuning
5. Evaluation (Accuracy, Precision, Recall)
6. Prediction on new text

## 📊 Key Observations
- Model showed higher confidence for "stress" class
- Possible reasons:
  - Class imbalance in dataset
  - Strong contextual signals in stress-related text
- Highlights importance of balanced data and evaluation metrics

## 📈 Results
- Accuracy: ~89%
- Improved performance compared to traditional ML models
- Better contextual understanding using transformers

## ⚠️ Limitations
- Sensitive to dataset bias
- Requires high computational resources
- Not suitable as a standalone mental health diagnostic tool

## 🔮 Future Improvements
- Use larger and more diverse datasets
- Apply techniques like class balancing
- Add explainability (SHAP/LIME)
- Deploy as a web app for real-time predictions

## 💡 Use Cases
- Mental health monitoring
- Social media sentiment analysis
- Early stress detection systems

