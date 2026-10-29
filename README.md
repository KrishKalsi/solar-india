# Solar India: rooftop solar forecasts for 22 Indian cities

Predicts tomorrow's solar energy (kWh/m²) and whether it will be a poor, average or good solar day,
from 11 years of NASA POWER satellite data and the Open-Meteo weather forecast.

**Live app:** _add your Streamlit link here_

## Run order
1. `notebooks/01_data.ipynb`: download NASA POWER data, clean it, time-based split
2. `notebooks/02_modelling.ipynb`: EDA, VIF, PCA, linear regression and neural network, logistic regression / LDA / KNN / SVM, final test, saves `models/`

## Run the app
```bash
pip install -r requirements.txt
streamlit run app.py
```
