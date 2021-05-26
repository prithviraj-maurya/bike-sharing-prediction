# inference.py
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import joblib

import config

################################################################
## Model's leaderboard scores:                                ##
## Decision Tree: score: 0.60506                              ##
################################################################

def handle_negative(preds):
    predictions = []
    for value in preds:
        if value < 0:
            predictions.append(0)
        else:
            predictions.append(value)
    return predictions


df_model_losses = pd.read_csv(config.MODEL_LOSSES_FILE)
df_test = pd.read_csv(config.TEST_FILE)

# plot all model's losses
print(df_model_losses)
plt.bar(df_model_losses['model'], df_model_losses['loss'])
# plt.show();

# choose the model with lowest loss
model_name = df_model_losses[df_model_losses['loss'] == df_model_losses['loss'].min()]['model'].values[0]
print(f"Loading model...{model_name}")
model = joblib.load(config.MODEL_OUTPUT + f"dt_{model_name}.bin")

print("making predictions...")
preds = model.predict(df_test.values)
predictions = handle_negative(preds)
submission = pd.read_csv('../input/sampleSubmission.csv')
submission['count'] = predictions
submission.to_csv(f"../output/submission_{model_name}.csv", index=False)    
print("Done submission")