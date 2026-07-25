AI-Based Flood Risk Forecasting — Indus Basin, Pakistan

A live, end-to-end AI system that predicts flood probability for the Indus Basin using real-time rainfall, weather, and river discharge data. Built from scratch — data collection, feature engineering, model training, and deployment.

🔗 Live dashboard: https://projects-xndqdkauixbdgh3azywren.streamlit.app

Project Goal

Predict the probability of flooding in the next 3 days using historical and live environmental data, and surface that prediction through a public, always-current dashboard — not a one-time static demo.

Why This Project

Most portfolio time-series projects predict stock or house prices. This one tackles a real operational problem: Pakistan has experienced major flooding in 2022, 2023, 2024, and 2025, with the Indus Basin at the center of it. An early-warning model here has genuine real-world value.

Data Sources
Source	What it provides	Access
NASA POWER API	Daily rainfall, temperature, humidity, wind speed (2020–present)	Free, no key required
Open-Meteo Flood API	Daily river discharge, based on the Copernicus GloFAS reanalysis	Free, no key required
NDMA 2022 Floods Dataset (Kaggle)	Regional damage/casualty context for the 2022 floods	Used for descriptive context only (no date column, not merged into training data)
Documented flood-event windows (2022–2025)	Manually compiled real flood date ranges	Used to build the flood/no-flood target label

Both live APIs mean the dataset naturally extends itself every time the notebook or dashboard is re-run — no manual re-downloading required.

Tech Stack
Python (Google Colab for development)
pandas / NumPy — data merging, cleaning, feature engineering
Plotly / Matplotlib — exploratory data analysis
scikit-learn — Random Forest, train/test splitting, evaluation metrics, RandomizedSearchCV
XGBoost and LightGBM — gradient boosting models
joblib — model serialization
Streamlit — the deployed dashboard
GitHub + Streamlit Community Cloud — hosting and deployment
Pipeline
Data Collection — pull rainfall/weather (NASA POWER) and river discharge (Open-Meteo) live; build flood labels from documented event windows
Preprocessing — merge all sources into one daily table, handle missing values with time-based interpolation, flag outliers
EDA — rainfall/discharge trends, flood frequency (class balance), seasonal patterns, correlation analysis
Feature Engineering:
Rainfall lag features (1, 3, 7 days)
Rolling rainfall sums (3, 7, 14, 21-day windows)
Rainfall intensity ratio
Consecutive rainy day streaks
River discharge rate-of-change (1, 3, 7-day deltas)
River discharge rolling mean/max (7, 14-day windows)
Cyclical month encoding (sin/cos) for seasonality
Modeling — binary classification (flood within next 3 days), time-based train/test split (never random, to avoid data leakage)
Model Results
Model	Recall	Precision	F1	ROC-AUC
Random Forest	0.70	0.52	0.60	0.80
LightGBM	0.69	0.48	0.57	0.79
XGBoost (tuned, deployed)	1.00	0.37	0.54	0.81

Why XGBoost, tuned for recall: in a flood early-warning system, missing a real flood is far more costly than a false alarm. The deployed model's decision threshold was lowered from the default 0.5 to 0.25, and hyperparameters were tuned via RandomizedSearchCV optimizing for recall/F2-score — deliberately trading some precision for the ability to catch every known flood event in testing.

What I Solved Along the Way
Started with a static Kaggle river dataset, caught that it only covered a single 2022 flood season — switched to the live Open-Meteo Flood API so the model could train on 2020–present instead of one event.
Debugged a live production bug: NASA POWER returns -999 as a placeholder for dates not yet fully processed (usually the most recent 1–3 days), which was corrupting the dashboard's rainfall chart and flood probability. Fixed by filtering the sentinel value before feature engineering.
Fixed a deployment error caused by uploading files into a GitHub folder with spaces in its name, which broke Streamlit Cloud's dependency parser — resolved by re-uploading files flat at the repo root.
Repository Structure
├── app.py                   # Streamlit dashboard (live inference)
├── requirements.txt         # Python dependencies for deployment
├── flood_model_final.pkl    # Trained XGBoost model + threshold + feature list
└── README.md
Running Locally
bash
pip install -r requirements.txt
streamlit run app.py
Dashboard Features
📍 Location selector (Tarbela, Mangla, Kalabagh gauge points)
📈 Live rainfall and river discharge charts
⚠️ Flood risk indicator (Low / Medium / High)
📅 7-day flood probability trend
📊 Model comparison table
Future Improvements
Add an LSTM/GRU sequence model for comparison against the tree-based baselines
Extend the flood-event label set beyond manually documented windows
Add satellite rainfall data for higher spatial resolution
SMS/email alerting for high-risk predictions
Data Attribution

Rainfall and weather data courtesy of NASA POWER. River discharge data courtesy of Open-Meteo, based on the Copernicus Global Flood Awareness System (GloFAS).