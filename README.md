# HealthAI Suite — Intelligent Analytics for Patient Care

## ⚙️ Workflow
1. Data Acquisition

Collected multi-domain healthcare data from hospital EHRs, ICU stay logs, lab panels, ED vital signs, chest X-rays, and doctor–patient chat records.

Each dataset was designed for a specific analytical task:

   - Risk stratification (classification)
   - Length of stay (regression)
   - Patient segmentation (clustering)
   - Comorbidity pattern detection (association rules)
   - Vitals forecasting (LSTM)
   - Imaging diagnostics (CNN)
   - Sentiment and chatbot analysis (NLP)

2. Data Cleaning

   - Removed duplicate and inconsistent records across datasets.
   - Imputed missing values using median for numerical and mode for categorical fields.
   - Handled outliers via IQR and Z-score trimming.
   - Normalized categorical values such as sex, admission type, and diagnosis group.
   - Time columns standardized for vitals (15-minute resampling).
   - Applied label encoding for disease classes and text normalization for chatbot data.

3. Feature Engineering

   - Scaled all numeric features using StandardScaler.
   - Performed Principal Component Analysis (PCA) to retain 95% variance.
   - Applied SMOTE for balancing imbalanced disease categories.
   - Derived new variables:
      - Comorbidity count
      - Temporal patterns (weekday, hour)
      - Vital sign trends (mean, slope, deviation)
   - Text data processed using TF-IDF vectorization and embedding models.
   - Images preprocessed (150×150 resize, grayscale normalization).

4. Model Building

   - Developed models for each module:
      - Classification: Random Forest, SVM, XGBoost.
      - Regression: Ridge(alpha=3.0), Ridge (log-target), HuberRegressor and RandomForest Regressor.
      - Clustering: HDBSCAN.
      - Association Rules: Apriori.
      - Time Series: LSTM (for vitals prediction).
      - Imaging: CNN with 5 convolutional layers.
      - NLP: TF-IDF with Logistic Regression.
   - Each model designed and tuned independently for its target dataset.
  
5. Model Evaluation

   - Used task-specific metrics for assessment:
      - Classification: Accuracy, Precision, Recall, F1-Score.
      - Regression: MAE, RMSE.
      - Clustering: Silhouette score.
      - Association Rules: Confidence, Lift.
      - Imaging: ROC-AUC, Accuracy.
      - Chatbot: BLEU score.
   - Applied cross-validation and 80/20 stratified split for consistency.
   - Achieved high stability and generalization across all model categories.
  
6. Results and Performance

   - Classification (Random Forest): ~91% accuracy.
   - Regression (Gradient Boosting): RMSE ≈ 1.6 days.
   - Clustering (HDBSCAN): Silhouette ≈ 0.62.
   - Association Rules (Apriori): Lift > 3 for key disease-treatment links.
   - Time Series (LSTM): MAE ≈ 3 bpm for heart-rate forecasts.
   - Imaging (CNN): 95% accuracy on X-ray diagnosis.
   - Chatbot (Logistic Regression): F1 ≈ 0.89, ensuring reliable sentiment prediction.
  
7. Streamlit Dashboard

   - Central interface combining all analytical modules.
   - Allows dataset uploads, exploration, and visualization.
   - Key sections:
      - EDA Dashboard: Missing values, correlations, feature distributions.
      - Risk Stratification: Predict disease likelihood.
      - Length of Stay: Estimate patient stay duration.
      - Clustering: View segmented patient groups.
      - Association Rules: Discover diagnosis–treatment patterns.
      - Time Series: Plot vital-sign forecasts (LSTM).
      - Imaging (CNN): Upload and classify X-rays.
      - Chatbot: Simulate multilingual doctor–patient dialogue.
    
## ▶️ Running the App

Ensure Python 3.9+ is installed.

1. Clone the repo:

       git clone https://github.com/Arjun-Karthik/HealthAI
       cd HealthAI

2.Install dependencies

       pip install -r requirements.txt

3. Run Streamlit app

       streamlit run healthai_streamlit_app.py

4. Upload CSV or image files when prompted to explore all modules.

## 🧩 Features

   - Modular ML pipelines across classification, regression, clustering, NLP, and CNN.
   - One-click visualization (Plotly, Matplotlib).
   - Model comparison dashboard.
   - Downloadable predictions and model artifacts (.pkl, .pt).
   - End-to-end explainability and interpretability tracking.

## ✅ Requirements

















