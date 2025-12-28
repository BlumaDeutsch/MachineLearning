from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
from ShowFunctions import show_heatmap

def run_model(model, X_train, X_test, y_train, y_test, params={}):
    start = time.time()
    result = model(**params)
    result.fit(X_train, y_train)
    runtime = time.time() - start
    predictions = calculate_predictions(result, X_test, y_test, average="macro")
    return {**predictions, 'runtime': runtime}


def calculate_predictions(m, X_test, y_test, average='binary'):
    # Make predictions on the test set
    y_pred = m.predict(X_test)

    # Evaluate accuracy
    Accuracy = accuracy_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred, average=average)
    Recall = recall_score(y_test, y_pred, average=average)
    F1 = f1_score(y_test, y_pred, average=average)
    AUC = roc_auc_score(y_test, m.predict_proba(X_test), multi_class='ovr')

    return {'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'F1': F1, 'AUC': AUC}


def processing_data(data):
    # הסרת עמודת ה-ID
    data = data.drop('ID', axis=1)

    # בדיקת חסרים
    print("Missing values per column:\n", data.isnull().sum())

    # בדיקת התפלגות המטרה
    target_counts = data['default.payment.next.month'].value_counts()
    print("\nTarget Distribution:\n", target_counts)

    show_heatmap(data)
