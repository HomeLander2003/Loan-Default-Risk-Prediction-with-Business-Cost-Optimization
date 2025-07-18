import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class Clean:
    def __init__(self):
        self.df = None

    def clean(self):
        try:
            file_path = r"D:\Bilal folder\internship\task4\credit_risk_dataset.csv"

            if os.path.isfile(file_path):
                self.df = pd.read_csv(file_path)
                print(self.df.head())
                print(self.df.info())
                print(np.shape(self.df))
                print(self.df.columns)

                column_drop = ["cb_person_cred_hist_length", "loan_grade"]
                for col in column_drop:
                    if col in self.df.columns:
                        self.df.drop(columns=col, inplace=True)
                        logging.info(f"Column '{col}' dropped")
                    else:
                        logging.warning(f"Column '{col}' not found in DataFrame")

                self.df.rename(columns={
                    "cb_person_default_on_file": "person_on_file",
                    "person_emp_length": "person_emp_year",
                    "loan_int_rate": "loan_rate"
                }, inplace=True)

                print("Initial rows:", self.df.shape[0])
                null_rows = self.df.isnull().any(axis=1).sum()
                print(f"Rows with nulls: {null_rows}")
                self.df.dropna(inplace=True)
                print("After dropna:", self.df.shape[0])

                duplicate_rows = self.df.duplicated().sum()
                print(f"Duplicate rows: {duplicate_rows}")
                self.df.drop_duplicates(keep="first", inplace=True)
                self.df.reset_index(drop=True, inplace=True)
                print("After drop_duplicates:", self.df.shape[0])
                
                   #visualizing outliers
                sns.boxplot(data=self.df, x=self.df["loan_rate"])
                plt.show()

                # Remove outliers
                Q1 = self.df['loan_rate'].quantile(0.25)
                Q3 = self.df['loan_rate'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                median = self.df['loan_rate'].median()
                self.df.loc[self.df['loan_rate'] > upper_bound, 'loan_rate'] = median
                self.df.loc[self.df['loan_rate'] < lower_bound, 'loan_rate'] = median
                
                sns.boxplot(data=self.df, x=self.df["loan_rate"])
                plt.show()

            else:
                logging.error("File path not correct or file not found.")

        except Exception as e:
            logging.error(f"Exception during file cleaning: {e}")

class Preprocessing(Clean):
    def prep(self):
        try:
            # Encode categorical columns
            self.df_encoded = pd.get_dummies(self.df, columns=["person_home_ownership", "loan_intent", "person_on_file"],
             drop_first=True, dtype=int)

            # Target = loan_status
            print("Target distribution:\n", self.df_encoded['loan_status'].value_counts())
            print(self.df_encoded.info())

        except Exception as e:
            logging.error(e)

class ML(Preprocessing):
    def ml(self):
        try:
            X = self.df_encoded.drop("loan_status", axis=1)
            y = self.df_encoded["loan_status"]

            x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)

            model = LogisticRegression(max_iter=1000)
            model.fit(x_train, y_train)
            probas = model.predict_proba(x_test)[:, 1]  # probability of class 1 (default)

            # Define business costs
            cost_fn = 500  # cost of false negative (predict non-default but actually defaulted)
            cost_fp = 100  # cost of false positive (predict default but actually didn't)

            best_threshold = 0.5
            best_cost = float('inf')

            for threshold in np.arange(0.1, 0.91, 0.01):
                preds = (probas >= threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
                total_cost = (cost_fp * fp) + (cost_fn * fn)

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_threshold = threshold

            # Final prediction with best threshold
            final_preds = (probas >= best_threshold).astype(int)

            print(f"\nBest Threshold: {best_threshold:.2f}")
            print(f"Minimum Business Cost: {best_cost}")
            print("\nAccuracy:", accuracy_score(y_test, final_preds))
            print("Classification Report:\n", classification_report(y_test, final_preds))

        except Exception as e:
            logging.error(e)

# Run everything
var = ML()
var.clean()
var.prep()
var.ml()
