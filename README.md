# Predicting Coronary Heart Disease (CHD)
Early identification of patients at risk of Coronary Heart Disease (CHD) can enable preventive care and follow-up diagnostics. 
Data source: train.csv (from a Kaggle dataset)

# Environment & Requirements
*Jupyte Notebook by Anaconda
Packages:
* pandas, numpy, scikit-learn
* imblearn (imbalanced-learn)
* matplotlib, seaborn (for plots)
* joblib (optional, for model persistence)
* sklearn

# Modeling pipeline
* Imputation: SimpleImputer
* Numeric: median
* Categorical: most frequent
* Encoding: OneHotEncoder(handle_unknown="ignore")
* Scaling: StandardScaler for numeric features
* Imbalance handling:
Primary: SMOTE (or SMOTEENN in alternative runs) applied within cross-validation folds on the training split only
Optionally combine with class_weight="balanced" in logistic regression
* Classifier: LogisticRegression (LBFGS / saga)
Tune C, penalty, and decision threshold (not just default 0.5) to target desired recall

# Evaluation:
Metrics: Recall, Precision, Accuracy, ROC–AUC

# Results & Interpretation

* Primary objective met: high sensitivity finds the majority of future CHD cases.
* Expected trade-off: lower precision (0.28) due to many false positives (187). Clinically tolerable since downstream diagnostics can filter these.
* Discrimination: ROC–AUC = 0.71 indicates fair separation; close to training (~0.73), so no strong overfitting signals.
* Accuracy (0.68): naturally drops when optimizing for recall under class imbalance; accuracy is less informative in this setting.
* Curves: ROC, Precision–Recall Confusion Matrix on independent test set
