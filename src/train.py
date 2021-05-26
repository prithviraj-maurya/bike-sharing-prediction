# train.py
import argparse
import os

import joblib
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

def run(model):
    # read data
    df_train = pd.read_csv(config.TRAINING_FILE)
    df_valid = pd.read_csv(config.VALID_FILE)
    df_model_losses = pd.read_csv(config.MODEL_LOSSES_FILE)

    # training data
    x_train = df_train.drop('count', axis=1).values
    y_train = df_train['count'].values

    # validation data
    x_valid = df_valid.drop('count', axis=1).values
    y_valid = df_valid['count'].values

    # model
    reg = model_dispatcher.models[model]

    # fit the model
    reg.fit(x_train, y_train)

    # predicitons
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str)

    args = parser.parse_args()
    run(model=args.model)
