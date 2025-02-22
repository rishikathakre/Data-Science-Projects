# Machine Learning Projects Collection

This repository contains multiple machine learning projects covering different domains, including price prediction, classification, and spam detection. Each project involves data preprocessing, exploratory data analysis (EDA), model building, and evaluation to derive meaningful insights and predictions.

## 📌 Projects Overview

### 1️⃣ Car Price Prediction 🚗💰

Objective: Predict the price of a car based on various features such as brand, model, year, mileage, fuel type, and transmission.

Techniques Used:

✔ Data Preprocessing (handling missing values, encoding categorical variables)

✔ Feature Engineering

✔ Regression Models (Linear Regression, Random Forest, XGBoost)

✔ Model Evaluation (R² Score, RMSE, MAE)

📂 File: Car_Price_Prediction.ipynb

### 2️⃣ Email Spam Detection 📩🚨


This project builds a machine learning-based email spam classifier to distinguish between spam and legitimate (ham) emails. Using Natural Language Processing (NLP) techniques, the text is cleaned, tokenized, and converted into numerical representations using TF-IDF and CountVectorizer.

Three machine learning models were tested:

Logistic Regression – 96.84% Accuracy

Multinomial Naïve Bayes – 96.69% Accuracy

Bernoulli Naïve Bayes – 98.63% Accuracy (Best Model)

The project successfully detects spam emails by analyzing patterns in word usage and frequency. It provides a highly accurate and efficient solution for filtering spam, making it useful for real-world email classification tasks. 🚀

Objective: Classify emails as spam or not spam

Techniques Used:

✔ Text Preprocessing (Tokenization, Lemmatization, Stopword Removal)

✔ TF-IDF Vectorization

✔ Machine Learning Models (Logistic Regression, Naïve Bayes, SVM)

✔ Model Evaluation (Accuracy, Precision, Recall, F1-Score)

📂 File: Email_Spam_Detection.ipynb

### 3️⃣ Iris Flower Classification 🌸🔬

Objective: Classify iris flowers into three species (Setosa, Versicolor, Virginica) based on sepal and petal dimensions.

Techniques Used:

✔ Exploratory Data Analysis (EDA)

✔ Feature Selection & Normalization

✔ Supervised Learning (KNN, Decision Trees, SVM)

✔ Model Evaluation (Confusion Matrix, Accuracy Score)

📂 File: Iris_Classification.ipynb

### 4️⃣ Sales Prediction 📊💵

Objective: Predict future sales based on historical sales data using machine learning models.

Techniques Used:

✔ Time Series Analysis

✔ Data Preprocessing (Handling Missing Data, Feature Engineering)

✔ Regression Models (Linear Regression, Decision Trees, Random Forest)

✔ Model Performance Metrics (MAPE, RMSE, MAE)

📂 File: Sales_prediction.ipynb

## 🛠️ How to Use the Projects

### 1️⃣ Clone the Repository

git clone https://github.com/rishikathakre/ml-projects.git

cd ml-projects

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Run the Notebooks

Open Jupyter Notebook or Google Colab

Load the respective .ipynb file

Execute the cells step by step

