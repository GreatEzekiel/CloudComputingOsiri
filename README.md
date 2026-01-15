Stock Transaction Predictor (CSIS505 Assessment)
🏢 Osiri University, Nebraska, USA
Course: Cloud Computing (CSIS505)

Project: Comparative Analysis of 23 ML Models for Stock Forecasting

Author: Adejare Ezekiel Sanyaolu

Submission Date: January 15, 2026

📌 Project Overview
This project presents a comprehensive machine learning pipeline to predict stock transaction types (BUY vs. SELL) using historical Yahoo Finance data. It evaluates 23 distinct algorithms to identify the most robust model for financial time-series classification, culminating in a cloud-deployed web application.

🚀 Live Application
Access the live cloud app here: 👉 https://cloudcomputingosiri-jddmpbbhcmmqtldvzf4tnr.streamlit.app/

🛠️ Technical Architecture
1. Data Source & Preprocessing
Dataset: Yahoo Finance Historical Transaction Data.

Features used: amount, reportedPrice, usdValue.

Handling Imbalance: Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the BUY/SELL classes, ensuring the model doesn't favor the majority class.

Scaling: Utilised StandardScaler to normalise feature distributions.

2. Algorithmic Rigour
The research compared 18 models across multiple categories:

Linear: Logistic Regression, Ridge, Lasso.

Tree-Based: Decision Trees, Extra Trees (Best Model), Random Forest.

Boosting: XGBoost, AdaBoost, Gradient Boosting, LightGBM.

Probabilistic/Other: Naïve Bayes, SVM, k-NN, MLP Neural Networks.

The Winner: The Extra Trees Classifier achieved a validation accuracy of ~72.47%, outperforming others in handling the high variance of financial data.

3. Cloud Stack
Language: Python 3.10+

Frontend/UI: Streamlit

Model Hosting: Streamlit Community Cloud

Version Control: GitHub

📂 Repository Structure
Plaintext

├── app.py                # Streamlit Web Application
├── train_best_model.py   # Training script & Pipeline
├── requirements.txt      # Cloud environment dependencies
├── best_model.pkl        # Serialized Extra Trees Model
├── scaler.pkl            # Serialized StandardScaler
├── yahooStock.csv        # Dataset (if permitted)
└── README.md             # Project Documentation
⚙️ Local Installation & Usage

Clone the repository:

Bash

git clone https://github.com/[Your-Username]/[Your-Repo-Name].git
cd [Your-Repo-Name]

Install dependencies:
pip install -r requirements.txt

Train the model (Optional):
python train_best_model.py

Launch the Web App:
streamlit run app.py

📊 Performance Metrics (Summary)
Metric	Result
Best Model	Extra Trees Classifier
Accuracy	72.47%
Technique	SMOTE + Standard Scaling
Deployment	PaaS (Streamlit Cloud)

Export to Sheets

📝 Academic Integrity
This assessment is submitted in partial fulfilment of the requirements for the Cloud Computing (CSIS505) course at Osiri University. All work, including the comparative analysis and application development, is the original work of the author.
