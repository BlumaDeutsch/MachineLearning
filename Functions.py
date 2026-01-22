from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from configs import *

def split_and_scale(data):
    X = data.drop('default.payment.next.month', axis=1)
    y = data['default.payment.next.month']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    numerical_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                      'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                      'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_test, y_train, y_test


def run_model(model, X_train, X_test, y_train, y_test, params={}):
    start = time.time()
    result = model(**params)
    result.fit(X_train, y_train)
    runtime = time.time() - start
    predictions = calculate_predictions(result, X_test, y_test)
    return {**predictions, 'Fit Time': runtime, 'Model': result}


def calculate_predictions(m, X_test, y_test, average='binary'):
    y_pred = m.predict(X_test)
    probs = m.predict_proba(X_test)
    Accuracy = accuracy_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred, average=average)
    Recall = recall_score(y_test, y_pred, average=average)
    F1 = f1_score(y_test, y_pred, average=average)
    AUC = roc_auc_score(y_test, probs[:, 1] if probs.shape[1] == 2 else probs, multi_class='ovr') # Supports both binary and multi-departmental case
    return {'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'F1': F1, 'AUC': AUC}


def processing_data(data):
    data = data.drop('ID', axis=1) # delete column id
    print("Missing values per column:\n", data.isnull().sum()) # check missing values
    target_counts = data['default.payment.next.month'].value_counts() # Checking the target distribution
    print("\nTarget Distribution:\n", target_counts)
    data['EDUCATION'] = data['EDUCATION'].replace([0, 5, 6], 4) # Handling the EDUCATION column Values 0, 5, 6 are considered undocumented, so we will merge them with 4 (Others)
    data['MARRIAGE'] = data['MARRIAGE'].replace(0, 3) # Handling the MARRIAGE column: Value 0 is not recorded, we will merge it with 3 (Others)
    data = pd.get_dummies(data, columns=['SEX', 'MARRIAGE', 'EDUCATION'], drop_first=True, dtype=int) # One-Hot Encoding
    print("\nNew columns after One-Hot Encoding:", data.columns)
    return data


def logistic_regression(X_train, X_test, y_train, y_test):
    return run_model(LogisticRegression, X_train, X_test, y_train, y_test)


def random_forest(X_train, X_test, y_train, y_test):
    all_results = []
    for cfg in forest_configs:
        result = run_model(RandomForestClassifier, X_train, X_test, y_train, y_test, cfg)
        result.update(cfg) # Adding the parameters to the results for the table
        all_results.append(result)
    return all_results


def adaboost(X_train, X_test, y_train, y_test):
    ada_results = []
    for cfg in ada_configs:
        result = run_model(AdaBoostClassifier, X_train, X_test, y_train, y_test, cfg)
        result.update(cfg) # Adding the parameters to the results for the table
        ada_results.append(result)
    return ada_results


def xgb(X_train, X_test, y_train, y_test):
    xgb_results = []
    for cfg in xgb_configs:
        result = run_model(XGBClassifier, X_train, X_test, y_train, y_test, cfg)
        result.update(cfg)
        xgb_results.append(result)
    return xgb_results
