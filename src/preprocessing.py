import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# loading the training data using pandas
df_train = pd.read_csv('../data/ECG5000/ECG5000_TRAIN.txt', header=None, delim_whitespace=True)
print(f'The original training data shape is: {df_train.shape}')
# print(df_train.head())

# loading the test data using pandas
df_test = pd.read_csv('../data/ECG5000/ECG5000_TEST.txt', delim_whitespace=True, header=None)
print(f'The original test data shape is: {df_test.shape}')
# print(df_test.head())

# merging the both datasets
df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
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
df_normal = df[df[0] == 1]
print(f'The shape of normal data is: {df_normal.shape}')
# print(df_normal.head())

# abnormal data with values == 2
df_abnormal2 = df[df[0] == 2]
print(f'The shape of abnormal_2 data is: {df_abnormal2.shape}')

# abnormal data with values == 3
df_abnormal3 = df[df[0] == 3]
print(f'The shape of abnormal_3 data is: {df_abnormal3.shape}')

# abnormal data with values == 4
df_abnormal4 = df[df[0] == 4]
print(f'The shape of abnormal_4 data is: {df_abnormal4.shape}')

# abnormal data with values == 5
df_abnormal5 = df[df[0] == 5]
print(f'The shape of abnormal_5 data is: {df_abnormal5.shape}')


def plot_sample(df_input, ax, title):
    data = df_input
    first_row = data.iloc[0, 1:]

    ax.plot(first_row.index, first_row.values, marker='o')
    ax.set_title(title)
    ax.set_xlabel('Sample Time')
    ax.set_ylabel('Values Obtained')
    ax.grid(True)


def plot_all_samples(df_1, df_2, df_3, df_4, df_5):
    fig, axs = plt.subplots(3, 2, figsize=(10, 12))

    plot_sample(df_1, axs[0, 0], 'First Row of Normal Sample')
    plot_sample(df_2, axs[1, 0], 'First Row of Abnormal Sample 2')
    plot_sample(df_3, axs[2, 0], 'First Row of Abnormal Sample 3')
    plot_sample(df_4, axs[0, 1], 'First Row of Abnormal Sample 4')
    plot_sample(df_5, axs[1, 1], 'First Row of Abnormal Sample 5')

    # Hide the empty subplot
    fig.delaxes(axs[2, 1])

    # Adjust layout
    plt.tight_layout()

    # Add spacing between subplots
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    plt.show()


# plot_all_samples(df_normal, df_abnormal2, df_abnormal3, df_abnormal4, df_abnormal5)


# concat all abnormal data
df_abnormal = pd.concat([df_abnormal2, df_abnormal3, df_abnormal4, df_abnormal5], ignore_index=True)
print(f'abnormal data shape is: {df_abnormal.shape}')

scaler = StandardScaler()

y_train = df_normal[0]
X_train = df_normal.drop(columns=[0], axis=1)

y_test = df_abnormal[0]
X_test = df_abnormal.drop(columns=[0], axis=1)

# Fit the scaler on the training data and transform both training and test data
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Convert back to DataFrame to print first few rows
X_train_normalized_df = pd.DataFrame(X_train_normalized, columns=X_train.columns)
X_test_normalized_df = pd.DataFrame(X_test_normalized, columns=X_test.columns)

# print(X_train.head())
# print(X_train_normalized_df.head())


X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_normalized, y_train, test_size=0.2,
                                                                          random_state=42)
print(f'X_train_split shape is : {X_train_split.shape} \n y_train_split shape is : {y_train_split.shape} \n '
      f'X_val_split shape is : {X_val_split.shape} \n y_val_split shape is : {y_val_split.shape} ')
