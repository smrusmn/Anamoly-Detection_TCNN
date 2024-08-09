from darts import TimeSeries
from darts.models import TCNModel
from darts.metrics import mape, mae
import torch


# Assuming the data is in suitable format and loaded into TimeSeries objects
train_series = TimeSeries.from_dataframe(pd.read_csv('../data/ECG5000/X_train_split.csv'))
val_series = TimeSeries.from_dataframe(pd.read_csv('../data/ECG5000/X_val_split.csv'))