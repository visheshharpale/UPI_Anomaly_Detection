# ATM/UPI Anomaly Detection System

## Overview

This project is a machine learning-based ATM and UPI transaction anomaly detection system. It identifies unusual banking transactions by learning normal transaction behaviour and detecting deviations.

The system combines Isolation Forest and an Autoencoder to generate anomaly scores.

## Features

- Synthetic banking transaction generation
- User behaviour-based feature engineering
- Isolation Forest anomaly detection
- Autoencoder-based anomaly detection
- Combined anomaly scoring
- Alert threshold selection
- Fraud pattern simulation
- Model evaluation using precision, recall, F1-score, ROC-AUC and PR-AUC
- Flask-based web application
- Transaction simulation and manual transaction APIs
- Anomaly explanation using transaction features

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Flask
- Matplotlib
- SciPy

## Project Structure

```text
ATM-UPI-Anomaly-Detection/
├── app.py
├── anomaly_core.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── templates/
│   └── index.html
└── plots/
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd ATM-UPI-Anomaly-Detection
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Research Pipeline

```bash
python main.py
```

The script trains the anomaly detection models, evaluates their performance on synthetic data and generates visualizations.

## Run the Web Application

```bash
python app.py
```

Open the application in your browser:
http://127.0.0.1:5000


## Models

### Isolation Forest

Detects unusual transactions by isolating observations that differ from normal behaviour.

### Autoencoder

Learns to reconstruct normal transaction patterns. Transactions with higher reconstruction errors are considered more anomalous.

### Combined Score

The system combines the percentile-based anomaly scores from both models.

## Fraud Patterns Simulated

- Unusually large transactions
- Transactions at unusual hours
- Sudden location changes
- Rapid transaction bursts

## Dataset

This project currently uses synthetic banking transaction data for demonstration and experimentation.

The generated labels are used only for evaluation and sanity checks. The anomaly detection models are trained without using fraud labels.

## Disclaimer

This is an academic demonstration project using synthetic data. It is not a production banking fraud detection system.

## Author

Vishesh Harpale, Rahul Takale, Yash Bhadale, Vedant Patil. 