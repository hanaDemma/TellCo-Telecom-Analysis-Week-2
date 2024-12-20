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


def get_outlier_summary(data):
    """
    Calculates outlier summary statistics for a DataFrame.

    Args:
        data : Input DataFrame.

    Returns:
        Outlier summary DataFrame.
    """

    outlier_summary = pd.DataFrame(columns=['Variable', 'Number of Outliers'])
    data = data.select_dtypes(include='number')

    for column_name in data.columns:
        q1 = data[column_name].quantile(0.25)
        q3 = data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]

        outlier_summary = pd.concat(
            [outlier_summary, pd.DataFrame({'Variable': [column_name], 'Number of Outliers': [outliers.shape[0]]})],
            ignore_index=True
        )
    non_zero_count = (outlier_summary['Number of Outliers'] > 0).sum()
    print(f"From {data.shape[1]} selected numerical columns, there are {non_zero_count} columns with outlier values.")

    return outlier_summary

def getBoxPlotToCheckOutlier(data, variables):
    for variable in variables:
        if pd.api.types.is_numeric_dtype(data[variable]):
            sns.boxplot(data=data, x=variable, orient='v')
            plt.title(f'Box Plot of {variable}')
            plt.xlabel('Values')
            plt.show()
        else:
            print(f"Skipping non-numeric column: {variable}")


def remove_outliers_winsorization(xdr_data):
    """
    Removes outliers from specified columns of a DataFrame using winsorization.

    Args:
        data: The input DataFrame.
        column_names (list): A list of column names to process.

    Returns:
        The DataFrame with outliers removed.
    """
    # data = xdr_data.select_dtypes(include='number')
    for column_name in xdr_data.select_dtypes(include='number').columns:
        q1 = xdr_data[column_name].quantile(0.25)
        q3 = xdr_data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        xdr_data[column_name] = xdr_data[column_name].clip(lower_bound, upper_bound)

    return xdr_data
