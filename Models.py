from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# from xgboost import XGBClassifier
import time


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
