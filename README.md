# Spotify Song Replay Prediction

A machine learning project that predicts whether a Spotify song will be **replayed** by a user, based on their listening behavior. Built as part of the Rhombix Technologies training program.

## 📌 Overview

This project analyzes user listening patterns to predict replay likelihood using a **Random Forest Classifier**. It covers the full ML pipeline — from raw data to actionable insights — and includes visualizations to interpret model behavior.

## 🎯 Problem Statement

Given a user's interaction with a song (how long they played it, whether they skipped it, whether shuffle was on), predict whether they are likely to **replay** that song.

## 🧠 Features Used

- **Play duration** — how long the song was played
- **Shuffle status** — whether shuffle mode was active
- **Skip behavior** — whether the user skipped the track

## ⚙️ Tech Stack

- **Python**
- **Pandas / NumPy** — data preprocessing
- **Scikit-learn** — Random Forest model training & evaluation
- **Matplotlib / Seaborn** — visualizations (feature importance, heatmaps)

## 🔄 Pipeline

1. **Data Preprocessing** — cleaning and preparing `spotify_history.csv`
2. **Feature Engineering** — extracting relevant behavioral signals
3. **Model Training** — Random Forest Classifier
4. **Evaluation** — accuracy and performance metrics
5. **Visualization** — feature importance chart, correlation heatmap

## 📂 Input / Output

| | |
|---|---|
| **Input** | `spotify_history.csv` |
| **Output** | Replay prediction (Yes/No) + behavioral insights |

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

## 📊 Results

The model identifies key behavioral signals (like skip rate and play duration) that most strongly influence whether a song gets replayed, visualized through a feature importance chart.

## 👩‍💻 Author

**Alishba Nadeem**
BS Artificial Intelligence, NUML Islamabad
[GitHub](https://github.com/Alishba-Nadeem1) · [LinkedIn](https://linkedin.com/in/alishba-nadeem-8014b9379)
