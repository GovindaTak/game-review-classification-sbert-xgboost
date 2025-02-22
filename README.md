# game-review-classification-sbert-xgboost
A project that classifies game reviews using Sentence Transformers (SBERT) and Machine Learning models. The first approach uses all-MiniLM-L6-v2 with Logistic Regression, and the second approach improves accuracy with paraphrase-mpnet-base-v2 and XGBoost.
------
# 🎮 Game Review Classification using Sentence Transformers & XGBoost

## 📌 Overview
This project aims to **classify game reviews** as **Positive (1) or Negative (0)** based on user suggestions using **Sentence Transformers (SBERT) for text embeddings** and **Machine Learning models** for classification.

### **🔥 Implemented Two Models:**
1️⃣ **SBERT (`all-MiniLM-L6-v2`) + Logistic Regression** → **82% Accuracy**  
2️⃣ **SBERT (`paraphrase-mpnet-base-v2`) + XGBoost** → **87% Accuracy**  

✅ **Uses pre-trained Transformer models for feature extraction**  
✅ **Works with raw text reviews without heavy preprocessing**  
✅ **Fast & scalable for large datasets**  

---

## 📊 Model Performance

| Model | Sentence Transformer Used | Classifier | Accuracy |
|--------|------------------------|------------|----------|
| **Baseline** | `all-MiniLM-L6-v2` | Logistic Regression | **82.22%** |
| **Optimized** | `paraphrase-mpnet-base-v2` | XGBoost | **87.00%** |

---

## 🚀 Installation & Setup

### **1️⃣ Install Dependencies**
Make sure you have **Python 3.9+** and install the required libraries:

pip install -U sentence-transformers xgboost scikit-learn pandas numpy


### **2️⃣ Load the Dataset**
If your dataset is stored in Google Drive, **mount it**:

from google.colab import drive
drive.mount('/content/drive')


✔ **Dataset Structure:**
- `"user_review"` → Contains the game review text.
- `"user_suggestion"` → The sentiment label (0 = Negative, 1 = Positive).

---

## 🛠️ Model Training & Evaluation

### **1️⃣ Load Pre-Trained Sentence Transformer**

from sentence_transformers import SentenceTransformer

# Load the optimized SBERT model
model = SentenceTransformer("paraphrase-mpnet-base-v2")


### **2️⃣ Encode Game Reviews into Embeddings**

import pandas as pd

# Load dataset
df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/datasets/game_review/train.csv")

# Convert reviews to embeddings
embeddings = model.encode(df["user_review"].tolist())


---

## **🎯 Model 1: Logistic Regression (`all-MiniLM-L6-v2`)**

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(embeddings, df["user_suggestion"], test_size=0.2, random_state=42)

# Train Logistic Regression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")

✔ **Achieved Accuracy:** `82.22%`

---

## **🔥 Model 2: XGBoost (`paraphrase-mpnet-base-v2`)**

from xgboost import XGBClassifier

# Train XGBoost classifier
xgb_clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, tree_method='gpu_hist')  # Enables GPU training
xgb_clf.fit(X_train, y_train)

# Evaluate XGBoost
y_pred_xgb = xgb_clf.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")

✔ **Achieved Accuracy:** `87.00%` 🎯

---

## **🔍 Finding Similar Reviews Using Cosine Similarity**
Use **cosine similarity** to find the most similar review to a given input.


from sentence_transformers import util
import numpy as np

query = "This game is fantastic!"
query_embedding = model.encode(query, device='cuda')  # Move to GPU

# Compute similarity with existing embeddings
similarities = util.cos_sim(query_embedding, embeddings)

# Get most similar review
most_similar_idx = np.argmax(similarities).item()
print(f"Most similar review: {df.iloc[most_similar_idx]['user_review']}")

✔ **This retrieves the most similar review from the dataset.**

---

## 📌 Key Features & Learnings
✔ **Sentence Transformers simplify text classification** by generating **dense vector embeddings**.  
✔ **SBERT (`paraphrase-mpnet-base-v2`) captures better contextual meaning** than `all-MiniLM-L6-v2`.  
✔ **XGBoost outperforms Logistic Regression in accuracy**.  
✔ **Finding similar reviews using cosine similarity can help in duplicate detection.**

---

## 📂 Folder Structure
```
📁 game-review-classification-sentence-transformer/
 ├── 📄 game_reviews.csv           # Dataset file
 ├── 📜 train_logistic_regression.py  # Baseline model (SBERT + Logistic Regression)
 ├── 📜 train_xgboost.py          # Optimized model (SBERT + XGBoost)
 ├── 📜 find_similar_reviews.py   # Code for similarity search
 ├── 📄 README.md                 # Project Documentation
```

---

## 💡 Future Improvements
🔹 Fine-tune `paraphrase-mpnet-base-v2` on gaming-specific reviews.  
🔹 Experiment with **LightGBM or CatBoost** for classification.  
🔹 Use **FAISS** for faster nearest neighbor search in large datasets.  
🔹 Apply **data augmentation** to handle class imbalance.  

---

## 📜 References
- [Sentence Transformers](https://www.sbert.net)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-Learn](https://scikit-learn.org/)

---

## 👨‍💻 Author & Contribution
Created by **Govinda Tak** 🎮🚀  
Feel free to contribute, raise issues, or suggest improvements!  

📌 **GitHub Repository Link:** 🔗   (https://github.com/GovindaTak/game-review-classification-sbert-xgboost.git)
---

### **📌 Summary**
✔ **Two approaches: Logistic Regression & XGBoost for classification.**  
✔ **Uses SBERT (`paraphrase-mpnet-base-v2`) for better embeddings.**  
✔ **Implemented similarity detection using cosine similarity.**  
✔ **Repository structured with scripts for easy execution.**  
