# 🎤 SentiSpeech: Speech Emotion Recognition using SVM

**SentiSpeech** is a machine learning project that detects and classifies human emotions from speech using audio feature extraction and a Support Vector Machine (SVM) classifier. Built on the RAVDESS dataset, the system analyzes vocal characteristics like pitch, tone, and energy to recognize emotions such as **happy**, **sad**, **angry** and more.

---

## 📁 Dataset

This project uses the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) audio-only speech subset:
- 24 professional actors (12 male, 12 female)
- 1440 recordings
- 8 emotions (each with normal & strong intensity)
- Neutral North American accent

For more details and download, visit: [RAVDESS on Zenodo](https://zenodo.org/record/1188976)

---

## 🔍 Features

- 🎧 Audio loading and preprocessing
- 📊 Feature extraction using Librosa (MFCC, Chroma, Spectral Contrast, etc.)
- 🧠 Emotion parsing from structured filenames
- 🧪 Model training with Support Vector Machine (SVM)
- 📈 Performance evaluation (accuracy, F1-score, confusion matrix)

---

## 🧠 Emotions Detected

- 😐 Neutral  
- 😊 Happy  
- 😢 Sad  
- 😠 Angry  
- 😲 Surprised  

---

