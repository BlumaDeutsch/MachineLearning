forest_configs = [
    {'n_estimators': 50,  'max_depth': 5,  'max_features': 'sqrt'},
    {'n_estimators': 100, 'max_depth': 10, 'max_features': 'log2'},
    {'n_estimators': 200, 'max_depth': 15, 'max_features': None}
]

ada_configs = [
    {'n_estimators': 50,  'learning_rate': 0.1},  # למידה איטית ושקולה
    {'n_estimators': 100, 'learning_rate': 1.0},  # קונפיגורציה סטנדרטית
    {'n_estimators': 500, 'learning_rate': 1.5}  # פוטנציאל גבוה ל-Overfitting
]

xgb_configs = [
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.7},
    {'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 10, 'subsample': 0.6}
]
