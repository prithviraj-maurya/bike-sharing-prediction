# model_dispatcher.py
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from lightgbm import LGBMRegressor

models = {
    'linear_regression': linear_model.LinearRegression(),
    'ridge': linear_model.Ridge(alpha=0.5),
    'svm': svm.SVR(),
    'decision_tree': tree.DecisionTreeRegressor(),
    'lgbm': LGBMRegressor()
}
