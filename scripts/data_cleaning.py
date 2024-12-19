import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def find_missing_values(df):
    """
    Finds missing values and returns a summary.

    Args:
        df: The DataFrame to check for missing values.

    Returns:
        A summary of missing values, including the number of missing values per column.
    """

    null_counts = df.isnull().sum()
    missing_value = null_counts
    percent_of_missing_value = 100 * null_counts / len(df)
    data_type=df.dtypes

    missing_data_summary = pd.concat([missing_value, percent_of_missing_value,data_type], axis=1)
    missing_data_summary_table = missing_data_summary.rename(columns={0:"Missing values", 1:"Percent of Total Values",2:"DataType" })
    missing_data_summary_table = missing_data_summary_table[missing_data_summary_table.iloc[:, 1] != 0].sort_values('Percent of Total Values', ascending=False).round(1)

    print(f"From {df.shape[1]} columns selected, there are {missing_data_summary_table.shape[0]} columns with missing values.")

    return missing_data_summary_table

def replace_missing_values(data):
    """
    Replaces missing values in a DataFrame with the mean for numeric columns 
    and the mode for categorical columns.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with missing values replaced.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    data_copy = data.copy()

    # Replace missing values in numeric columns with the mean
    for column in data_copy.select_dtypes(include='number').columns:
        column_mean = data_copy[column].mean()
        data_copy[column].fillna(column_mean, inplace=True)

    # Replace missing values in categorical columns with the mode
    for column in data_copy.select_dtypes(include='object').columns:
        if not data_copy[column].mode().empty:  # Check if mode exists
            column_mode = data_copy[column].mode().iloc[0]
            data_copy[column].fillna(column_mode, inplace=True)

    return data_copy
