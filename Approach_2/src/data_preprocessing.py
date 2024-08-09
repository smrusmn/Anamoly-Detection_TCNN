import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler


# Load the data (replace 'train.txt' and 'test.txt' with your actual file names)
train_data = pd.read_csv('../data/ECG5000_TRAIN.txt', delim_whitespace=True, header=None)
test_data = pd.read_csv('../data/ECG5000_TEST.txt', delim_whitespace=True, header=None)


# Check for null values in both datasets
print("Null values in training data:", train_data.isnull().sum().sum())
print("Null values in testing data:", test_data.isnull().sum().sum())




# Merge the datasets row-wise
combined_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

train_final = combined_data[combined_data[0] == 1].reset_index(drop=True)
test_final = combined_data[combined_data[0] != 1].reset_index(drop=True)

# Drop the label column (0th column)
train_final = train_final.drop(columns=[0])
test_final = test_final.drop(columns=[0])

# Convert to TimeSeries objects
series = TimeSeries.from_dataframe(train_final)
test_series = TimeSeries.from_dataframe(test_final)

# Manually split the data into train and validation sets (e.g., 80% train, 20% val)
train_size = int(0.8 * len(series))
train_series = series[:train_size]
val_series = series[train_size:]






# Normalize the data using Darts Scaler
scaler = Scaler()
train_series_scaled = scaler.fit_transform(train_series)
val_series_scaled = scaler.transform(val_series)
test_series_scaled = scaler.transform(test_series)