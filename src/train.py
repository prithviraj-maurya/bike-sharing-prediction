# train.py
import argparse
import os
from functools import partial

import joblib
import optuna
import pandas as pd
from sklearn import metrics

import config
import model_dispatcher

def handle_negative(preds):
    predictions = []
    for value in preds:
        if value < 0:
            predictions.append(0)
        else:
            predictions.append(value)
    return predictions

def get_training_data():
    # read data
    df_train = pd.read_csv(config.TRAINING_FILE)
    df_valid = pd.read_csv(config.VALID_FILE)

    # training data
    x_train = df_train.drop('count', axis=1).values
    y_train = df_train['count'].values

    # validation data
    x_valid = df_valid.drop('count', axis=1).values
    y_valid = df_valid['count'].values

    return x_train, y_train, x_valid, y_valid

def run(model):
    df_model_losses = pd.read_csv(config.MODEL_LOSSES_FILE)
    x_train, y_train, x_valid, y_valid = get_training_data()

    # model
    reg = model_dispatcher.models[model]

    # fit the model
    print("fitting the data....")
    reg.fit(x_train, y_train)

    # predicitons
    print("making prediction....")
    predictions = reg.predict(x_valid)
    preds = handle_negative(predictions)

    # calculate and print loss
    loss = metrics.mean_squared_log_error(preds, y_valid)
    print(f"RMSLE loss {loss}")

    # save loss to output loss file
    model_losses = pd.DataFrame({ 'model': [model], 'loss': [loss]})
    df_model_losses = df_model_losses.append(model_losses)
    df_model_losses.to_csv(config.MODEL_LOSSES_FILE, index=False)

    # save the model
    joblib.dump(reg, os.path.join(config.MODEL_OUTPUT, f"dt_{model}.bin"))


## Tune hyperparameters
def objective(trial, model_name):
    x_train, y_train, x_valid, y_valid = get_training_data()
     # 2. Suggest values of the hyperparameters using a trial object.
    param = model_dispatcher.get_model_params(trial=trial, model_name=model_name)
    
    reg = model_dispatcher.models[model_name]
    reg.set_params(**param)
    print(reg)
    reg.fit(x_train, y_train)
    # predicitons
    predictions = reg.predict(x_valid)
    preds = handle_negative(predictions)

    # calculate and print loss
    loss = metrics.mean_squared_log_error(preds, y_valid)
    return loss

def optimize_model(model_name):
    study = optuna.create_study(direction="minimize")
    objective_func = partial(objective, model_name=model_name)
    study.optimize(objective_func, n_trials=100)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)

    args = parser.parse_args()
    run(model=args.model)
    optimize_model(model_name=args.model)