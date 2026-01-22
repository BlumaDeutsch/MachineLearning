import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import scipy.stats as stats

def show_distribution(data, diag):
    # נניח שזה ה-DataFrame שלך והעמודה נקראת 'weight'
    # קודם כל, נבודד רק את הערכים הקיימים (בלי ה-NaN)
    weights = data[diag].dropna()

    # 1. ויזואליזציה - נצייר היסטוגרמה כדי לראות את ה"פעמון"
    plt.figure(figsize=(8, 5))
    plt.hist(weights, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Weight Distribution (Non-Missing Values)')
    plt.xlabel(diag)
    plt.ylabel('Frequency')
    plt.savefig(diag + '.png')

    # 2. מבחן שפירו-וילק לנורמליות
    # H0: הנתונים מתפלגים נורמלית
    stat, p_value = stats.shapiro(weights)

    print(f"Shapiro-Wilk Test: Statistics={stat:.3f}, p-value={p_value:.3f}")

    if p_value > 0.05:
        #print("הנתונים נראים נורמליים (לא ניתן לדחות את הנחת הנורמליות)")
        return True
    else:
        #print("הנתונים לא מתפלגים נורמלית")
        return False

def show_heatmap(data, target_col, threshold=0.05):
    # חישוב קורלציה
    corr = data.corr(numeric_only=True)

    # סינון: נשאיר רק משתנים שיש להם קשר מינימלי למטרה
    relevant_features = corr[target_col][abs(corr[target_col]) > threshold].index
    filtered_corr = data[relevant_features].corr()

    plt.figure(figsize=(12, 10))
    # square=True מבטיח שהמשבצות יהיו ריבועיות
    sns.heatmap(filtered_corr, annot=False, cmap='coolwarm', fmt=".2f", square=True)
    plt.title(f'Features with Correlation > {threshold} to {target_col}')
    plt.show()
    # plt.figure(figsize=(8, 10))
    # sns.heatmap(data.corr(numeric_only=True), annot=False, cmap='coolwarm', linewidths=0.5)
    # plt.title(title)
    # plt.savefig(f'{filename}.png', bbox_inches='tight')
    # plt.show()


def show_comparison_table(tabl_data, configs):
    comparison_table = pd.DataFrame(tabl_data)
    comparison_table = comparison_table.drop(columns=['Model'])
    # Arrange the columns so that the parameters are at the beginning
    cols = list(configs[0].keys()) + [c for c in comparison_table.columns if c not in configs[0].keys()]
    comparison_table = comparison_table[cols]
    print(comparison_table)


def show_comparison_graph(graph_data):
    comparison_table = pd.DataFrame(graph_data)
    comparison_df = comparison_table.sort_values(by='n_estimators')
    x_values = comparison_df['n_estimators']
    y_values = comparison_df['F1']
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker='o', linestyle='-', color='teal', markersize=8)
    plt.title('F1 Score vs Number of Estimators (From Collected Results)')
    plt.xlabel('Number of Trees (n_estimators)')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def show_results(logistic_result, forest_results, ada_results, xgb_results):
    final_summary_data = [
        {**logistic_result, 'Model': 'Logistic Regression'},
        max(forest_results, key=lambda x: x['AUC']),
        max(ada_results, key=lambda x: x['AUC']),
        max(xgb_results, key=lambda x: x['AUC'])
    ]
    for i, name in enumerate(['Logistic Regression', 'Random Forest', 'AdaBoost', 'XGBoost']):
        final_summary_data[i]['Model Name'] = name
    summary_df = pd.DataFrame(final_summary_data)
    cols_order = ['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Fit Time']
    summary_df = summary_df[cols_order]
    print(summary_df)


def show_confusion_matrix(model, X_test, y_test, title):
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=['No Readmit', 'Readmitted'],
        cmap='Blues',
        values_format='d')
    plt.title("Confusion Matrix - " + title)
    plt.show()


def show_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-10:]  # take the 10 that is most important
    plt.figure(figsize=(10, 8))
    plt.title('Top 10 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='#3498db', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance - ' + title)
    plt.show()