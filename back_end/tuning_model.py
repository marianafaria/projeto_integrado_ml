from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import lightgbm as lgb
import xgboost as xgb

def get_best_model(X_train, y_train):
    algoritmos = {
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100, 200, 250, 600],
                'max_depth': [None, 3, 4, 5, 10, 20, 30],
                'min_samples_split': [2, 4, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy'],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced'],
                'oob_score': [True, False]
            }
        },
        'GradientBoostingClassifier': {
            'model': GradientBoostingClassifier(),
            'params': {
                'n_estimators': [50, 150, 250],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.7, 0.8, 0.9],
                'max_features': ['sqrt', 'log2'],
                'loss': ['exponential'],
            }
        },
        'XGBClassifier': {
            'model': xgb.XGBClassifier(),
            'params': {
                'n_estimators': [50, 150, 250],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 2, 4],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'gamma': [0, 0.1, 0.2],
                'reg_alpha': [0, 0.1, 0.2],
                'reg_lambda': [0, 0.1, 0.2],
                'objective': ['binary:logistic'],
                'eval_metric': ['rmse'],
            }
        },
        'LGBMClassifier': {
            'model': lgb.LGBMClassifier(),
            'params': {
                'n_estimators': [50, 150, 250],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
            }
        }
    }

    best_model = None
    best_score = 0

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    for algoritmos_name, config in algoritmos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv)
        gs.fit(X_train, y_train)

        if gs.best_score_ > best_score:
            best_model = gs.best_estimator_
            best_score = gs.best_score_

    return best_model
