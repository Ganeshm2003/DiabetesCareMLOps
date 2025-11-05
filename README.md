# DiabetesCare MLOps 🏥🤖

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Enabled-blue)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)

## 🎯 Project Overview

DiabetesCare MLOps is a cutting-edge machine learning system that predicts hospital readmission risk for diabetes patients. Built with modern MLOps practices, it ensures reliable, reproducible, and production-ready predictions.

## ✨ Key Features

- 🔮 **Predictive Analytics**: Advanced ML models to predict hospital readmission risk
- 📊 **MLflow Integration**: Complete experiment tracking and model versioning
- 🌐 **Streamlit Web App**: User-friendly interface for real-time predictions
- 🛠️ **Production-Ready**: Robust MLOps pipeline with preprocessing and evaluation
- 📈 **Performance Metrics**: ROC-AUC and PR-AUC metrics for model evaluation

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/alricdsouza11/DiabetesCareMLOps.git
   cd DiabetesCareMLOps
   ```

2. **Set up the environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app_streamlit.py
   ```

## 🏗️ Project Structure

```
├── app_streamlit.py          # Streamlit web application
├── requirements.txt          # Project dependencies
├── data/                    # Dataset directory
│   ├── diabetes.csv         # Main dataset
│   └── IDS_mapping.csv      # ID mapping file
├── src/                     # Source code
│   ├── data_preprocessing.py # Data preparation
│   ├── train.py            # Model training
│   ├── evaluate.py         # Model evaluation
│   ├── inference.py        # Prediction pipeline
│   └── utils.py            # Utility functions
└── mlruns/                 # MLflow experiment tracking
```

## 📊 Model Performance

Our model is evaluated using key metrics:
- ROC-AUC Score
- PR-AUC Score (Precision-Recall)

Track model performance and experiments through MLflow's intuitive interface.

## 🛠️ Technical Stack

- **ML Framework**: scikit-learn
- **Experiment Tracking**: MLflow
- **Web Interface**: Streamlit
- **Data Processing**: pandas, numpy
- **Version Control**: Git

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or enhancements
- Submit pull requests
- Improve documentation
- Share feedback

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaboration, feel free to reach out or open an issue.

---
*Built with ❤️ for better healthcare predictions*
