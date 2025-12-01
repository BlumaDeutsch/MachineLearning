from Models import run_model
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def main():
    data = load_wine()
    x = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    params = {'n_estimators': 100, 'max_depth': 5, 'max_features': 'sqrt'}
    result = run_model(RandomForestClassifier, X_train, X_test, y_train, y_test, params)
    print("result: ", result)


if __name__ == '__main__':
    main()

