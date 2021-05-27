# model_dispatcher.py
import xgboost as xgb
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor

models = {
    'linear_regression': linear_model.LinearRegression(),
    'ridge': linear_model.Ridge(alpha=0.5),
    'svm': svm.SVR(),
    'decision_tree': tree.DecisionTreeRegressor(),
    'gbr': GradientBoostingRegressor(),
    'lgbm': LGBMRegressor(),
    'xgb': xgb.XGBRegressor(),
    'lgbm-optimized': LGBMRegressor(
        lambda_l1= 1.2442475967572832,
        lambda_l2= 1.0232775643501282e-06,
        num_leaves= 216,
        feature_fraction= 0.998953748626822,
        bagging_fraction= 0.9014682923487376,
        bagging_freq= 3,
        min_child_samples= 5,
    )
}


def get_model_params(trial, model_name):
    param = {}
    if 'lgbm' in model_name:
        param = {
            # 'objective': 'binary',
            # 'metric': 'binary_logloss',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }

    return param