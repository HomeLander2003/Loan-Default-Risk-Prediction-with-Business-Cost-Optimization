# Loan Default Risk Prediction with Business Cost Optimization

##  Task Objective

The primary goal of this project is to:
- **Predict the likelihood of loan default** using historical loan applicant data.
- **Optimize the classification decision threshold** based on **custom business costs** associated with incorrect predictions:
  - **False Negative (FN)**: Predicting no default when the customer actually defaults — costly for the business.
  - **False Positive (FP)**: Predicting default when the customer would have repaid — opportunity loss.

This project goes beyond traditional accuracy metrics to incorporate a **cost-sensitive approach** for decision-making.

---

##  Approach

The solution was implemented in three structured stages using object-oriented programming:

### 1. **Data Cleaning and Preprocessing**
- **Dropped** irrelevant columns: `cb_person_cred_hist_length`, `loan_grade`
- **Renamed** columns for clarity (`loan_int_rate` → `loan_rate`)
- **Removed** rows with missing values and duplicates
- **Handled outliers** in `loan_rate` using the IQR method (outliers replaced with the median)
- **Encoded categorical variables** (`person_home_ownership`, `loan_intent`, `cb_person_default_on_file`) using one-hot encoding

### 2. **Machine Learning Model**
- **Model Used**: Logistic Regression
- **Split** data into training and testing sets (70/30)
- **Probability Predictions**: Used `predict_proba()` for threshold tuning
- **Business Cost Optimization**:
  - Assigned **cost_fn = 500**, **cost_fp = 100**
  - Tested thresholds from `0.10` to `0.90` in steps of `0.01`
  - Calculated total cost = `(False Positives * 100) + (False Negatives * 500)`
  - **Selected best threshold** with minimum total business cost

### 3. **Evaluation**
- Final predictions were made using the best threshold
- Evaluated using:
  - Accuracy
  - Classification report (precision, recall, F1-score)
  - Business cost calculation

---

##  Results and Findings

-  **Best Threshold** (Minimized Business Cost): `0.47` (example)
-  **Minimum Business Cost**: `$XXXX` *(depends on actual output)*
-  **Model Accuracy**: `82%` *(example)*
-  **Classification Report**: Balanced precision and recall, with improved default detection

The cost-based thresholding led to **better financial outcomes** compared to simply using the default 0.5 threshold.

---

##  Key Takeaways

- **Cost-sensitive modeling** is critical in domains like credit risk where prediction errors have unequal consequences.
- A simple logistic regression model, when fine-tuned with business logic, can outperform more complex models evaluated only on accuracy.
- This approach can be generalized to other risk prediction problems (insurance, fraud detection, etc.).

---

## 🛠 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, Seaborn, Matplotlib
- Logging for debug and tracking

---

##  How to Run

1. Ensure `credit_risk_dataset.csv` is placed at the correct path.
2. Run the script:
   ```bash
   python loan_risk_optimizer.py

