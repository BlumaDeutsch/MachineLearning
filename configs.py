forest_configs = [
    {'n_estimators': 50, 'max_depth': 5, 'max_features': 'sqrt', 'class_weight': 'balanced'},
    {'n_estimators': 100, 'max_depth': 10, 'max_features': 'log2', 'class_weight': 'balanced'},
    {'n_estimators': 200, 'max_depth': 12, 'max_features': 'sqrt', 'class_weight': 'balanced'}
]

ada_configs = [
    {'n_estimators': 50, 'learning_rate': 0.1},  # למידה איטית ושקולה
    {'n_estimators': 100, 'learning_rate': 1.0},  # קונפיגורציה סטנדרטית
    {'n_estimators': 500, 'learning_rate': 1.5}  # פוטנציאל גבוה ל-Overfitting
]

xgb_configs = [
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8, 'scale_pos_weight': 1.5},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.7, 'scale_pos_weight': 1.5},
    {'n_estimators': 500, 'learning_rate': 0.01, 'max_depth': 10, 'subsample': 0.6, 'scale_pos_weight': 1.5}
]

demographics = ['age', 'gender', 'race', 'readmitted']

clinical_metrics = [
    'time_in_hospital', 'num_lab_procedures', 'num_procedures',
    'num_medications', 'number_diagnoses', 'max_glu_serum',
    'A1Cresult', 'readmitted'
]

medications = [
    'insulin', 'metformin', 'glipizide', 'glyburide',
    'pioglitazone', 'rosiglitazone', 'change', 'diabetesMed', 'readmitted'
]

diags = [
    'diag_1', 'diag_2', 'diag_3'
]
