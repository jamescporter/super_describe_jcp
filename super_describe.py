import pandas as pd
import numpy as np
import pyarrow #Added to suppress PyCharm warning. Not Familiar with the pandas 3 documentation yet.


def super_describe(df):
    """Takes a pandas DataFrame as an input and returns a transposed DataFrame with the calculated mean, median,
    standard deviation, variation, skewness, coefficient of variation as percentage, mean/median difference,
    and kurtosis for each numeric column."""

    # Select only numeric columns from the DataFrame
    df_numeric = df.select_dtypes(include=[np.number])

    # Calculate statistical measures
    means = round(df_numeric.mean(), 2)
    medians = df_numeric.median()
    std_devs = round(df_numeric.std(), 2)
    variations = round(df_numeric.var(), 2)
    skew = df_numeric.skew()
    co_of_v = (df_numeric.std() / df_numeric.mean()) * 100
    mean_med_dif = df_numeric.mean() - df_numeric.median()
    kurt = df_numeric.kurtosis()

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Mean': means,
        'Median': medians,
        'Standard Deviation': std_devs,
        'Variation': variations,
        'Skewness': skew,
        'Coefficient of Variation %': co_of_v,
        'Mean-Median Difference': mean_med_dif,
        'Kurtosis': kurt
    })

    return summary_df.T

path = #YOUR FILE PATH HERE

df = pd.read_csv(path)

print(super_describe(df))
