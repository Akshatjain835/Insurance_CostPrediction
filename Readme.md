# Health Insurance Cost Prediction

A machine learning web application that predicts health insurance payment amounts based on demographic and health-related features. The application is built using Streamlit and employs various regression models to provide accurate insurance cost estimates.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Data Features](#data-features)
- [Contributing](#contributing)

## ✨ Features

- Interactive web interface built with Streamlit
- Real-time insurance cost prediction
- Support for multiple input features:
  - Age
  - Gender
  - BMI (Body Mass Index)
  - Blood Pressure
  - Number of children
  - Diabetic status
  - Smoking status
- Pre-trained machine learning model with high accuracy
- User-friendly form-based input system

## 📁 Project Structure

```
Insurance prediction/
│
├── data/
│   └── insurance1.csv          # Dataset used for training
│
├── encoders/
│   ├── label_encoder_gender.pkl
│   ├── label_encoder_diabetic.pkl
│   └── label_encoder_smoker.pkl
│
├── models/
│   └── best_model.pkl          # Trained model (XGBoost)
│
├── notebooks/
│   └── Insurance_cost_prediction1.ipynb  # Jupyter notebook for EDA and model training
│
├── src/
│   └── app.py                  # Streamlit application
│
├── scaler.pkl                  # StandardScaler for feature normalization
└── README.md                   # Project documentation
```

## 🛠 Technologies Used

- **Python 3.x**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Joblib** - Model serialization
- **Matplotlib/Seaborn** - Data visualization (in notebook)

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Akshatjain835/Insurance_CostPrediction.git
   cd "Insurance prediction"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install streamlit pandas numpy scikit-learn xgboost joblib
   ```

## 🚀 Usage

1. **Ensure all model files are present**
   - `scaler.pkl` (in root directory)
   - `models/best_model.pkl`
   - `encoders/label_encoder_gender.pkl`
   - `encoders/label_encoder_diabetic.pkl`
   - `encoders/label_encoder_smoker.pkl`

2. **Run the Streamlit application**
   ```bash
   streamlit run src/app.py
   ```

3. **Access the application**
   - The app will open in your default web browser
   - Default URL: `http://localhost:8501`

4. **Make a prediction**
   - Fill in the required information in the form
   - Click "Predict payment" button
   - View the estimated insurance payment amount

## 📊 Model Performance

The project explores multiple regression models:

| Model | R² Score | MAE | RMSE |
|-------|----------|-----|------|
| **XGBoost** | 0.81 | 4023.99 | 5484.56 |
| Random Forest | 0.81 | 4158.22 | 5539.31 |
| Polynomial Regression | 0.77 | 4527.93 | 6094.63 |
| Linear Regression | 0.71 | 5233.21 | 6854.09 |
| SVR | 0.48 | 6434.99 | 9208.17 |

**Best Model**: XGBoost (R² = 0.81) is used as the final model for predictions.

### Model Hyperparameters (XGBoost)
- Learning Rate: 0.05
- Max Depth: 3
- N Estimators: 100
- Subsample: 1.0

## 📈 Data Features

The model uses the following features for prediction:

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| Age | Numerical | Age of the individual | 0-100 |
| Gender | Categorical | Gender of the individual | Male/Female |
| BMI | Numerical | Body Mass Index | 10.0-60.0 |
| Blood Pressure | Numerical | Blood pressure reading | 60-200 |
| Children | Numerical | Number of dependents | 0-8 |
| Diabetic | Categorical | Diabetic status | Yes/No |
| Smoker | Categorical | Smoking status | Yes/No |

**Target Variable**: `claim` - Insurance payment amount (USD)

## 🔧 Model Training

To retrain the model or explore the data:

1. Open the Jupyter notebook: `notebooks/Insurance_cost_prediction1.ipynb`
2. The notebook includes:
   - Exploratory Data Analysis (EDA)
   - Data preprocessing
   - Feature engineering
   - Model training and evaluation
   - Hyperparameter tuning using GridSearchCV
   - Model comparison and selection

## 📝 Notes

- The application uses pre-trained models. To update predictions, retrain the model using the Jupyter notebook.
- Ensure all encoder files are compatible with the trained model.
- The scaler and encoders must be the same ones used during model training.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

---

**Note**: This application is for educational and demonstration purposes. Actual insurance pricing involves many additional factors and should be calculated by certified insurance professionals.

