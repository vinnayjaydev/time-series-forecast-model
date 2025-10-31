# 🌐 **AI AVENGERS PROMPT MAP — FDE REVENUE FORECASTING PROJECT**

---

## 🧩 **STEP 1 — PROJECT BLUEPRINT (Claude)**
**Goal:** Create the folder structure, Makefile, and README.  

**Prompt for Claude:**
```
Create a complete folder structure, README, and Makefile for a project titled
“Future Revenue Forecasting using Historical and Textual Data (FDE Project)”.

Project Description:
A Python-based machine learning project that forecasts future monthly revenue using
historical financial data and textual line-item descriptions.
Uses NLP, clustering, XGBoost, and time series forecasting to improve accuracy.

Deliverables:
• revenue_forecast_model.pkl
• preprocessing_pipeline.pkl
• model_metrics.json
• feature_importance.png
• cluster_analysis.csv
• monthly_forecasts.csv
• retrain_script.py

Include these directories:
- data/raw, data/processed
- src (with modules: preprocess.py, feature_engineering.py, clustering.py, model_training.py, time_series_model.py, evaluate.py, retrain_script.py, dashboard/app.py)
- models/
- reports/
- requirements.txt
- README.md

Makefile commands:
- make preprocess
- make train
- make evaluate
- make retrain
- make dashboard
```

---

## 📊 **STEP 2 — DATA GENERATION (Gemini)**
**Goal:** Generate synthetic 24-month dataset with textual line items.  

**Prompt for Gemini:**
```
Generate a synthetic dataset named revenue_data.csv with 24 months of data (Jan 2024 – Dec 2025).
Include around 300–500 rows.

Columns:
- month (YYYY-MM)
- region (North, South, East, West)
- product_category (Software, Hardware, Services)
- line_item_description (short invoice-like text)
- revenue (float between 10,000–200,000)

Ensure realistic variation and save as CSV.
Show first 10 rows of the dataset preview.
```

---

## 🧠 **STEP 3 — PREPROCESSING & NLP PIPELINE (ChatGPT)**
**Goal:** Clean and transform data, handle text and categorical features.  

**Prompt for ChatGPT:**
```
Generate src/preprocess.py for the FDE Revenue Forecasting Project.

Steps:
1. Load revenue_data.csv
2. Handle missing values
3. Encode categorical variables (region, product_category)
4. Tokenize 'line_item_description' using NLTK
5. Remove stopwords, apply TF-IDF vectorization
6. Combine structured + textual features
7. Save preprocessing_pipeline.pkl and clean_data.csv

Use sklearn.pipeline and joblib for saving.
Ensure clean modular design and logging for each stage.
```

---

## 🔍 **STEP 4 — CLUSTERING (ChatGPT)**
**Goal:** Add cluster-based feature for interpretability.  

**Prompt for ChatGPT:**
```
Generate src/clustering.py for the FDE project.

Steps:
1. Load clean_data.csv and TF-IDF matrix
2. Apply KMeans clustering to the textual TF-IDF features
3. Determine optimal clusters using Elbow Method (optional)
4. Add 'cluster_label' column to dataset
5. Save cluster_analysis.csv
```

---

## ⚙️ **STEP 5 — MODEL TRAINING (ChatGPT)**
**Goal:** Build and train XGBoost model.  

**Prompt for ChatGPT:**
```
Generate src/model_training.py for the FDE Revenue Forecasting Project.

Steps:
1. Load processed data with cluster features
2. Split into train/test (80/20)
3. Train XGBoostRegressor
4. Optimize hyperparameters using RandomizedSearchCV
5. Evaluate using RMSE, MAE, and R²
6. Save model_metrics.json
7. Plot and save feature importance as feature_importance.png
8. Save trained model as revenue_forecast_model.pkl
```

---

## ⏳ **STEP 6 — TIME SERIES FORECASTING (ChatGPT)**
**Goal:** Forecast next 3 months using ARIMA or Prophet.  

**Prompt for ChatGPT:**
```
Create src/time_series_model.py that trains a time series model for revenue forecasting.

Steps:
1. Aggregate revenue by month from clean_data.csv
2. Use Prophet or ARIMA for forecasting
3. Predict next 3 months revenue
4. Plot historical + forecast data
5. Save outputs as monthly_forecasts.csv and revenue_forecast_plot.png
```

---

## ⚙️ **STEP 7 — AUTOMATED RETRAINING (ChatGPT)**
**Goal:** Automate periodic model updates.  

**Prompt for ChatGPT:**
```
Generate src/retrain_script.py that automates retraining for the FDE project.

Functionality:
- Load new monthly data
- Update preprocessing and clustering
- Retrain XGBoost + time series models
- Save new models and metrics
- Integrate with Makefile (make retrain)
- Include logging and versioning of models
```

---

## 📈 **STEP 8 — DASHBOARD INTEGRATION (Gemini or ChatGPT)**
**Goal:** Interactive visualization and model insights.  

**Prompt (Gemini if Streamlit, ChatGPT if Plotly):**
```
Build src/dashboard/app.py using Streamlit.

Features:
- Upload new data and trigger retraining
- Display revenue trend (historical vs forecast)
- Show feature importance chart
- Show cluster segmentation visualization
- Display model performance metrics (RMSE, MAE, R²)
Add buttons: "Retrain Model", "Refresh Forecast"
```

---

## 🪮 **STEP 9 — MATH & FORECAST VALIDATION (DeepSeek)**
**Goal:** Verify your math and statistical integrity.  

**Prompt for DeepSeek:**
```
Review my forecasting approach in model_training.py and time_series_model.py.
Check if RMSE, MAE, and R² are implemented correctly, and whether Prophet/ARIMA
is used properly for monthly data. Suggest improvements for statistical validity.
```

---

## 🖥️ **STEP 10 — LOCAL DEPLOYMENT (ChatGPT)**
**Goal:** Run locally and deploy to IBM Watson.  

**Prompt for ChatGPT:**
```
Provide full local deployment instructions for the FDE project.

Include:
1. Creating virtual environment
2. Installing dependencies
3. Running Makefile commands
4. Launching dashboard locally
5. Packaging for IBM Watson Studio deployment

Also include sample README content for easy documentation.
```

---

# ⚡ BONUS PROMPTS

### 🪮 For **DeepSeek** (sanity check your results):
> Validate the correlation between time series residuals and forecasted revenue. Are they autocorrelated? If yes, suggest model improvement.

### 💡 For **Gemini** (EDA visuals):
> Create 3 plots: revenue trend over time, region-wise average revenue, and product-category distribution. Use Matplotlib and Seaborn.

---

# 📂 INPUT DATA SCHEMA
| Column | Type | Description |
|--------|------|-------------|
| month | string | Format YYYY-MM |
| region | string | Region name (North/South/East/West) |
| product_category | string | Product category (Software/Hardware/Services) |
| line_item_description | string | Text invoice-like description |
| revenue | float | Monthly revenue value |

---

# 🧠 FINAL TL;DR
- **Claude** → Project structure + docs  
- **Gemini** → Generate dataset + visuals + dashboard  
- **ChatGPT (GPT-5)** → Preprocessing + clustering + modeling + retraining  
- **DeepSeek** → Validate your math & forecasts  
- **Copilot** → Clean up code in VS Code  
- **Grok** → Crack a joke when you rage-debug 😅

