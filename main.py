from Functions import processing_data, split_and_scale, xgb, adaboost, random_forest, logistic_regression
import pandas as pd
from ShowFunctions import show_comparison_table, show_comparison_graph, show_heatmap, show_results, \
    show_confusion_matrix, show_feature_importance
from configs import *


def main():
    data = pd.read_csv("UCI_Credit_Card.csv")
    print("First 5 rows:\n", data.head())
    data = processing_data(data)
    show_heatmap(data)
    X_train, X_test, y_train, y_test = split_and_scale(data)
    logistic_result = logistic_regression( X_train, X_test, y_train, y_test)
    print(logistic_result)
    forest_results = random_forest(X_train, X_test, y_train, y_test)
    show_comparison_table(forest_results, forest_configs)
    show_comparison_graph(forest_results)
    ada_results = adaboost(X_train, X_test, y_train, y_test)
    show_comparison_table(ada_results, ada_configs)
    xgb_results = xgb(X_train, X_test, y_train, y_test)
    show_comparison_table(xgb_results, xgb_configs)
    show_results(logistic_result, forest_results, ada_results, xgb_results)
    show_confusion_matrix(logistic_result['Model'], X_test, y_test, 'Logistic Regression')
    show_confusion_matrix(max(forest_results, key=lambda x: x['AUC'])['Model'], X_test, y_test, 'Random Forest')
    show_confusion_matrix(max(ada_results, key=lambda x: x['AUC'])['Model'], X_test, y_test, 'AdaBoost')
    show_confusion_matrix(max(xgb_results, key=lambda x: x['AUC'])['Model'], X_test, y_test, 'XGBoost')
    show_feature_importance(max(forest_results, key=lambda x: x['AUC'])['Model'], X_train.columns, 'Random Forest')
    show_feature_importance(max(xgb_results, key=lambda x: x['AUC'])['Model'], X_train.columns, 'XGBoost')


if __name__ == '__main__':
    main()
