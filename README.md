# Online Payments Fraud Detection
This project uses machine learning to detect fraudulent online payment transactions. It involves data exploration, feature engineering, and the comparison of three different classification models: Logistic Regression, Decision Tree, and Random Forest.
## About the Project
The goal of this notebook is to build a model that can accurately predict fraudulent transactions from a large dataset of online payments. The process includes:

- Loading and exploring the dataset.
- Cleaning the data by filtering for relevant transaction types ('TRANSFER' and 'PAYMENT').
- Engineering new features to better capture transaction patterns.
- Training and evaluating three different ML models to find the best performer.
## Dataset
The data is from the "Online Payments Fraud Detection Dataset" available on Kaggle:

- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset)
- In this notebook, the dataset is loaded from the file upi\_dataset.csv.
## **Project Workflow**
1. **Data Loading & Exploration:** The dataset is loaded into a Pandas DataFrame and its shape, columns, and basic statistics are examined.
1. **Data Preprocessing:**
   1. Transactions are filtered to keep only PAYMENT and TRANSFER types, as these are the only types where fraud occurs in this dataset.
   1. The data is checked for null values and duplicates.
1. **Feature Engineering:** New features are created to improve model performance:
   1. time\_hr: The hour of the day (extracted from the step feature).
   1. is\_night: A binary flag for transactions occurring at night (before 6 AM).
   1. sender\_balance\_change & receiver\_balance\_change: The change in account balances post-transaction.
   1. dest\_balance\_zero & orig\_balance\_zero: Binary flags for when the pre-transaction balance was zero.
   1. The type column is one-hot encoded.
1. **Visualization:** A correlation heatmap is generated using Seaborn to understand relationships between numerical features.
1. **Model Training & Evaluation:** The data is split into training and testing sets. Three models are trained and evaluated:
   1. **Logistic Regression** (with StandardScaler)
   1. **Decision Tree Classifier** (with class\_weight to handle imbalance)
   1. **Random Forest Classifier** (with class\_weight)
## **Results**
All models performed well, but the **Random Forest Classifier** achieved the best results, correctly identifying fraudulent transactions with near-perfect precision and recall on the test set.
### **Best Model: Random Forest Classifier**

|**Metric**|**Score**|
| :- | :- |
|**Accuracy**|**99.998%**|
|**Precision (Class 1 - Fraud)**|0\.99|
|**Recall (Class 1 - Fraud)**|1\.00|
|**F1-Score (Class 1 - Fraud)**|0\.99|

**Confusion Matrix:**

``[[670047      7]``
``[4         1043]]``
## **Technologies Used**
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, StandardScaler, train\_test\_split, metrics)
- Matplotlib & Seaborn
## **How to Run**
1. Clone this repository.
1. Install the required libraries:\
   ``pip install pandas numpy scikit-learn matplotlib seaborn jupyter``
1. Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset) and save it as upi\_dataset.csv in the root directory.
1. Open and run the main.ipynb notebook.
