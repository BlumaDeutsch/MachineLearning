from Functions import processing_data, split_and_scale, xgb, adaboost, random_forest, logistic_regression, predict_new_data
import pandas as pd
from ShowFunctions import show_comparison_table, show_comparison_graph, show_heatmap, show_results, \
    show_confusion_matrix, show_feature_importance
from configs import *


def main():
    data = pd.read_csv("health_student_data.csv", na_values='?') # replace ? to empty cells
    print("First 5 rows:\n", data.head())
    best_model, train_cols = processing_data(data)

    test_data = pd.read_csv("health_final_exam_input.csv", na_values='?')
    predict_new_data(test_data, best_model, train_cols)


if __name__ == '__main__':
    main()
