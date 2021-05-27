# inference.py
import pandas as pd
import joblib

import config

################################################################
## Model's leaderboard scores:                                ##
## Decision Tree: score: 0.60506                              ##
## Light GBM: score: 0.52300                                  ##
## Light GBM Optimized: score: 0.49073                        ##
## Voting Regressor: score: 0.48968                           ##
## Voting Regressor Featured(Datetime): score: 0.54566        ##
################################################################

def handle_negative(preds):
    predictions = []
    for value in preds:
        if value < 0:
            predictions.append(0)
        else:
            predictions.append(value)
    return predictions

# read losses file and get all models
df_model_losses = pd.read_csv(config.MODEL_LOSSES_FILE)
df_test = pd.read_csv('../input/test_featured.csv')
# df_test = pd.read_csv(config.TEST_FILE)
print(df_model_losses)

# choose the model with lowest loss
model_name = df_model_losses[df_model_losses['loss'] == df_model_losses['loss'].min()]['model'].values[0]
print(f"Loading model...{model_name}")
model = joblib.load(config.MODEL_OUTPUT + f"dt_{model_name}.bin")


## Submission
print("making submission...")
preds = model.predict(df_test.values)
predictions = handle_negative(preds)
submission = pd.read_csv('../input/sampleSubmission.csv')
submission['count'] = predictions
submission.to_csv(f"../output/submission_{model_name}.csv", index=False)    
print("Done submission")