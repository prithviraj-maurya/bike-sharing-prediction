## imports
import pandas as pd

## read data
dir = '../input/'
train_processed = pd.read_csv(dir + 'train_processed.csv')
test_processed = pd.read_csv(dir + 'test_processed.csv')
train_normalised = pd.read_csv(dir + 'train_normalised.csv')
test_normalised = pd.read_csv(dir + 'test_normalised.csv')
train_featured = pd.read_csv(dir + 'train_featured.csv')
test_featured = pd.read_csv(dir + 'test_featured.csv')

normalise = False
featured = True

if normalise:
    train = train_normalised.copy()
    test = test_normalised.copy()
if featured:
    train = train_featured.copy()
    test = test_featured.copy()    
else:
    train = train_processed.copy()
    test = test_processed.copy()

print(f"Train {train.shape}")
print(f"Test {test.shape}")

## For splitting data we should follow same approach as train and test data do
## For every month of every year:
## Day 1 - Day 16 will be training data
## Day 16 - Day 19 (training data has till date 19) will be validation data
train_fold = pd.DataFrame(columns=train.columns)
valid_fold = pd.DataFrame(columns=train.columns)
years = [2011, 2012]
months = list(range(1, 13))

for year in years:
    for month in months:
        train_data = train[(train['year'] == year) & (train['month'] == month) & (train['day'] <= 16)]
        valid_data = train[(train['year'] == year) & (train['month'] == month) & (train['day'] > 16)]
        train_fold = train_fold.append(train_data, ignore_index=True)
        valid_fold = valid_fold.append(valid_data, ignore_index=True)

print(f"train fold {train_fold.shape}")
print(train_fold.head())
print(f"valid fold {valid_fold.shape}")
print(valid_fold.head())

print(f"Percentage of validation data = {((valid_fold.shape[0] / train.shape[0]) * 100):.2f}%")

# normalised data doesn't work here as the day column is normalised 
# and will have to find that value to compare
## export fold dataframes as csv
# if normalise:
#     train_fold.to_csv(dir + 'train_fold_normalised.csv', index=False)
#     valid_fold.to_csv(dir + 'valid_fold_normalised.csv', index=False)
# else:
#     train_fold.to_csv(dir + 'train_fold.csv', index=False)
#     valid_fold.to_csv(dir + 'valid_fold.csv', index=False)

train_fold.to_csv(dir + 'train_fold.csv', index=False)
valid_fold.to_csv(dir + 'valid_fold.csv', index=False)

print("exported to csv..")    