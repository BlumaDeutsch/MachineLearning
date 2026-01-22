from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from configs import *
import random
from ShowFunctions import show_comparison_table, show_comparison_graph, show_heatmap, show_results, \
    show_confusion_matrix, show_feature_importance, show_distribution


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
    AUC = roc_auc_score(y_test, probs[:, 1] if probs.shape[1] == 2 else probs,
                        multi_class='ovr')  # Supports both binary and multi-departmental case
    return {'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'F1': F1, 'AUC': AUC}


def processing_data(data):
    # print("Missing values per column:\n", (data.isnull().sum() / len(data)) * 100)  # check missing values
    data = delete_columns_with_uniform_value(data)
    data = delete_columns_with_random_value(data)
    data = delete_columns_with_Shortages(data)
    data = categorical_to_boolean(data)
    data = ordinal_encoding(data)
    data = define_medical_category(data)
    data = fill_and_format_medical_tests(data)

    # print("Missing values per column:\n", (data.isnull().sum() / len(data)) * 100)  # check missing values

    X_train, X_test, y_train, y_test = split_data(data)

    X_train, X_test = fill_with_frequent(X_train, X_test)
    X_train, X_test = fill_with_default(X_train, X_test)
    X_train, X_test = one_hot_encoding(X_train, X_test)

    X_train, X_test = scale(X_train, X_test)
    #show_heatmap(pd.concat([X_train, y_train], axis=1), 'readmitted')

    logistic_result = logistic_regression(X_train, X_test, y_train, y_test)
    print(logistic_result)

    X_train, X_test = delete_columns_with_small_threshold(X_train, X_test, logistic_result)

    forest_results = random_forest(X_train, X_test, y_train, y_test)
    show_comparison_table(forest_results, forest_configs)
    show_confusion_matrix(forest_results[2]['Model'], X_test, y_test, 'random_forest')

    xgb_results = xgb(X_train, X_test, y_train, y_test)
    show_comparison_table(xgb_results, xgb_configs)
    show_confusion_matrix(xgb_results[1]['Model'], X_test, y_test, 'xgb')

    return forest_results[2]['Model'], X_train.columns



def scale(X_train, X_test):
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_test

def delete_columns_with_small_threshold(X_train, X_test, result):
    threshold = 0.0005
    importance = pd.Series(abs(result['Model'].coef_[0]), index=X_train.columns).sort_values()
    cols_to_drop = importance[importance < threshold].index

    X_train = X_train.drop(columns=cols_to_drop)
    X_test = X_test.drop(columns=cols_to_drop)
    #print(f"Removed {len(cols_to_drop)} weak features: ", cols_to_drop)
    return X_train, X_test


def delete_columns_with_uniform_value(data):
    cols_to_drop = ['glipizide-metformin', 'citoglipton', 'examide', 'metformin-pioglitazone']
    data = data.drop(columns=cols_to_drop)
    return data


def delete_columns_with_random_value(data):
    cols_to_drop = ['encounter_id', 'patient_nbr']
    data = data.drop(columns=cols_to_drop)
    return data


def delete_columns_with_Shortages(data):
    data = data.drop(columns=['weight', 'payer_code'])
    return data


def categorical_to_boolean(data):
    if 'readmitted' in data.columns:
        data['readmitted'] = (data['readmitted'] == '>30')
    unknown_mask = data['gender'] == 'Unknown/Invalid'
    data.loc[unknown_mask, 'gender'] = np.random.choice([True, False], size=unknown_mask.sum())

    gender_map = {'Male': False, 'Female': True}
    data['gender'] = data['gender'].map(gender_map).fillna(data['gender'])  # שומר על ה-True/False מההגרלה
    # data['gender'] = data['gender'].map({'Male': False, 'Female': True, 'Unknown/Invalid': random.choice([True, False])})
    cols_to_fix = ['diabetesMed', 'change', 'troglitazone', 'tolbutamide', 'metformin-rosiglitazone', 'acetohexamide',
                   'glimepiride-pioglitazone']
    for col in cols_to_fix:
        data[col] = (data[col] != 'No')
    return data


def ordinal_encoding(data):
    age_mapping = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, '[50-60)': 5, '[60-70)': 6,
                   '[70-80)': 7,
                   '[80-90)': 8, '[90-100)': 9, '[100-110)': 10, '[110-120)': 11}
    data['age'] = data['age'].map(age_mapping)
    return data


def fill_and_format_medical_tests(data):
    data['max_glu_serum'] = (data['max_glu_serum'] != 'Norm') & (data['max_glu_serum'].notna())
    data['A1Cresult'] = (data['A1Cresult'] != 'Norm') & (data['A1Cresult'].notna())
    return data


def define_medical_category(data):
    for col in ['diag_1', 'diag_2', 'diag_3']:
        data[col] = data[col].apply(map_icd9_to_category)
    return data


def split_data(data):
    X = data.drop('readmitted', axis=1)
    y = data['readmitted']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def fill_with_frequent(X_train, X_test):
    frequent_imputer = SimpleImputer(strategy='most_frequent')
    X_train[['race']] = frequent_imputer.fit_transform(X_train[['race']])
    X_test[['race']] = frequent_imputer.transform(X_test[['race']])
    return X_train, X_test


def fill_with_default(X_train, X_test):
    fill_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
    X_train[['medical_specialty']] = fill_imputer.fit_transform(X_train[['medical_specialty']])
    X_test[['medical_specialty']] = fill_imputer.transform(X_test[['medical_specialty']])
    return X_train, X_test


def one_hot_encoding(X_train, X_test):
    cols_to_fix = ['race', 'diag_1', 'diag_2', 'diag_3', 'medical_specialty', 'glyburide-metformin', 'insulin',
                   'miglitol', 'rosiglitazone', 'acarbose', 'pioglitazone', 'glyburide', 'glimepiride', 'glipizide',
                   'nateglinide', 'chlorpropamide', 'repaglinide', 'metformin', 'tolazamide']
    X_train = pd.get_dummies(X_train, columns=cols_to_fix, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cols_to_fix, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test


def map_icd9_to_category(value):
    try:
        code = float(value)
        if 390 <= code <= 459 or code == 785:
            return 'Circulatory'
        elif 460 <= code <= 519 or code == 786:
            return 'Respiratory'
        elif 520 <= code <= 579 or code == 787:
            return 'Digestive'
        elif 250 <= code < 251:
            return 'Diabetes'
        elif 800 <= code <= 999:
            return 'Injury'
        elif 710 <= code <= 739:
            return 'Musculoskeletal'
        elif 580 <= code <= 629 or code == 788:
            return 'Genitourinary'
        elif 140 <= code <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'  # starts with V or E


def logistic_regression(X_train, X_test, y_train, y_test):
    return run_model(LogisticRegression, X_train, X_test, y_train, y_test)


def random_forest(X_train, X_test, y_train, y_test):
    all_results = []
    for cfg in forest_configs:
        result = run_model(RandomForestClassifier, X_train, X_test, y_train, y_test, cfg)
        result.update(cfg)  # Adding the parameters to the results for the table
        all_results.append(result)
    return all_results


def adaboost(X_train, X_test, y_train, y_test):
    ada_results = []
    for cfg in ada_configs:
        result = run_model(AdaBoostClassifier, X_train, X_test, y_train, y_test, cfg)
        result.update(cfg)  # Adding the parameters to the results for the table
        ada_results.append(result)
    return ada_results


def xgb(X_train, X_test, y_train, y_test):
    xgb_results = []
    for cfg in xgb_configs:
        result = run_model(XGBClassifier, X_train, X_test, y_train, y_test, cfg)
        result.update(cfg)
        xgb_results.append(result)
    return xgb_results


def predict_new_data(test_data, model, train_columns):
    test_data = categorical_to_boolean(test_data)
    test_data = ordinal_encoding(test_data)
    test_data = define_medical_category(test_data)
    test_data = fill_and_format_medical_tests(test_data)

    X_new = pd.get_dummies(test_data)

    X_new = X_new.reindex(columns=train_columns, fill_value=0)

    predictions = model.predict(X_new)
    print(f"Number of patients predicted to be readmitted: {sum(predictions)}")
    return predictions
