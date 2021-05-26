import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import train_test_split

#########################################
## Baseline score: 1.35453 (processed) ##
## Baseline score: 1.35261 (normalise) ##
## Sample Submission score: 4.76188    ##
#########################################

## utility functions
## For negative predictions turn them to 0
def handle_negative(preds):
    predictions = []
    for value in preds:
        if value < 0:
            predictions.append(0)
        else:
            predictions.append(value)
    return predictions

## Read data
normalise = True
train_df = pd.read_csv('../input/train_processed.csv')
test_df = pd.read_csv('../input/test_processed.csv')
train_normalised = pd.read_csv('../input/train_normalised.csv')
test_normalised = pd.read_csv('../input/test_normalised.csv')

## check if to use normalize data or not
if normalise:
    train = train_normalised.copy()
    test = test_normalised.copy()
else:
    train = train_df.copy()
    test = test_df.copy()

print(f"Normalised data: {normalise}")
print(f"Train {train.shape}")
print(f"Test {test.shape}")
print("------------------------------")

np.random.seed(42)
X = train.drop('count', axis=1).values
y = train['count'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = LinearRegression(normalize=True)
reg.fit(X_train, y_train)

preds = reg.predict(X_test)        
predictions = handle_negative(preds)


## Predictions    
loss = mean_squared_log_error(predictions, y_test)
score = reg.score(X_test, y_test)
print(f"Loss {loss:.4f}")
print(f"score {score}")
print("Prediciton completed!! starting submission.....")

## Submissions
preds = reg.predict(test.values)
predictions = handle_negative(preds)
submission = pd.read_csv('../input/sampleSubmission.csv')
submission['count'] = predictions
if normalise:
    submission.to_csv('../output/baseline_submission_normalised.csv', index=False)
else:
    submission.to_csv('../output/baseline_submission_processed.csv', index=False)    
print("Done submission")