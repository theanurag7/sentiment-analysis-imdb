# Sentiment Analysis System

Binary sentiment classification on 50,000 IMDB movie reviews 
using an ensemble of 5 ML models, exposed via ASP.NET Core Web API.

## 🎯 Accuracy
| Model | Accuracy |
|-------|----------|
| Naive Bayes | 86.95% |
| Logistic Regression | 90.14% |
| SVM (LinearSVC) | 89.63% |
| Random Forest | 85.26% |
| **Voting Classifier** | **90.29% ⭐** |

## 🏗️ Architecture
User Input → ASP.NET Core Web API → Python ML Model → Positive/Negative

## 🛠️ Tech Stack
- **ML:** Python, Scikit-learn, TF-IDF, Voting Classifier
- **API:** ASP.NET Core Web API (C#)
- **Dataset:** IMDB 50K Movie Reviews

## 📁 Project Structure
- `sentimentanalysis.py` — Main ML training code
- `save_model.py` — Saves trained model to disk
- `predict_only.py` — Loads model and predicts sentiment
- `SentimentController.cs` — .NET API endpoint
- `Program.cs` — .NET app startup
