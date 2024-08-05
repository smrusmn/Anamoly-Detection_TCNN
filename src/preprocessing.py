import pandas as pd
import matplotlib.pyplot as plt


# loading the training data using pandas
df_train = pd.read_csv('../data/ECG5000/ECG5000_TRAIN.txt', header=None, delim_whitespace=True)
print(f'The original training data shape is: {df_train.shape}')
# print(df_train.head())

# loading the test data using pandas
df_test = pd.read_csv('../data/ECG5000/ECG5000_TEST.txt', delim_whitespace=True, header=None)
print(f'The original test data shape is: {df_test.shape}')
# print(df_test.head())

# merging the both datasets
df = pd.concat([df_train, df_test],axis=0, ignore_index=True)
print(f'The merged data shape now is: {df.shape}')
# print(df.head())

# Checking for null values
null_values = df.isnull().sum()
print(f'Null values in each column:\n{null_values}')

# Checking for duplicate rows
duplicate_rows = df.duplicated().sum()
print(f'Number of duplicate rows: {duplicate_rows}')

# Print the rows that are duplicated (if any)
if duplicate_rows > 0:
    print(f'Duplicate rows:\n{df[df.duplicated()]}')

# Basic statistics of the dataset
print(f'Statistics of given data: {df.describe()}')

# Checking the distribution of the target variable (assuming the first column is the target)
target_distribution = df.iloc[:, 0].value_counts()
print(f'Distribution of target variable:\n{target_distribution}')

# Dividing into normal and abnormal

# firstly, normal data with values == 1
df_normal1 = df[df[0]==1]
print(f'The shape of normal data is: {df_normal1.shape}')

# abnormal data with values == 2
df_abnormal2 = df[df[0]==2]
print(f'The shape of abnormal_2 data is: {df_abnormal2.shape}')

# abnormal data with values == 3
df_abnormal3 = df[df[0]==3]
print(f'The shape of abnormal_3 data is: {df_abnormal3.shape}')

# abnormal data with values == 4
df_abnormal4 = df[df[0]==4]
print(f'The shape of abnormal_4 data is: {df_abnormal4.shape}')

# abnormal data with values == 5
df_abnormal5 = df[df[0]==5]
print(f'The shape of abnormal_5 data is: {df_abnormal5.shape}')

def plot_sample(df_input):
    df = df_input
    first_row = df.iloc[0, :-1]


    plt.figure(figsize=(8, 4))
    plt.plot(first_row.index, first_row.values, marker='o')
    plt.title(f'First Row of the Dataset, Label = {first_row[0]}')
    plt.xlabel('Sample Time')
    plt.ylabel('Values Obtained')
    plt.grid(True)
    plt.show()


plot_sample(df_normal1)
plot_sample(df_abnormal2)
plot_sample(df_abnormal3)
plot_sample(df_abnormal4)
plot_sample(df_abnormal5)


# concat all abnormal data
df_abnormal = pd.concat([df_abnormal2, df_abnormal3, df_abnormal4, df_abnormal5], ignore_index=True)
print(f'abnormal data shape is: {df_abnormal.shape}')

