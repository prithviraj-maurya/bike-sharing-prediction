import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import train_test_split


## Read data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print(f"Train {train.shape}")
print(f"Test {test.shape}")

train['year'] = pd.to_datetime(train.datetime).dt.year
train['month'] = pd.to_datetime(train.datetime).dt.month
train['day'] = pd.to_datetime(train.datetime).dt.day
train['hour'] = pd.to_datetime(train.datetime).dt.hour
test['year'] = pd.to_datetime(test.datetime).dt.year
test['month'] = pd.to_datetime(test.datetime).dt.month
test['day'] = pd.to_datetime(test.datetime).dt.day
test['hour'] = pd.to_datetime(test.datetime).dt.hour

# preprocessing


X = train.drop(['datetime', 'casual', 'registered', 'count'], axis=1).values
y = train['count'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = LinearRegression(normalize=True)
reg.fit(X_train, y_train)

preds = reg.predict(X_test)
predictions = []
## For negative predictions
for value in preds:
    if value < 0:
        predictions.append(0)
    else:
        predictions.append(value)    

## Predictions        
loss = mean_squared_log_error(predictions, y_test)
score = reg.score(X_test, y_test)
print(f"Loss {loss:.4f}")
print(f"score {score}")
predictions = reg.predict(test.drop('datetime', axis=1).values)
submission = pd.read_csv('./data/sampleSubmission.csv')
print(len(predictions), submission.shape)
