import pandas as pd
import numpy as np
from scipy.stats import hmean, gmean, trim_mean, median_abs_deviation


def count_outliers_tukey(series):
    """Count outliers using Tukey's fences (1.5*IQR rule)."""
    clean_series = series.dropna()
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = ((clean_series < lower_fence) | (clean_series > upper_fence)).sum()
    return outliers

def count_outliers_1p96_sd(series, mean, std):
    """Count outliers using 1.96 * standard deviation rule with precomputed mean and std."""
    clean_series = series.dropna()
    lower_fence = mean - 1.96 * std
    upper_fence = mean + 1.96 * std
    outliers = ((clean_series < lower_fence) | (clean_series > upper_fence)).sum()
    return outliers


def super_describe(df):
    """
    Generate a comprehensive set of statistical measures for each numeric column in a DataFrame.

    Parameters:
    df (pd.DataFrame): Input pandas DataFrame.

    Returns:
    pd.DataFrame: DataFrame with statistical measures.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Select only numeric columns from the DataFrame and drop rows with NaNs
    df_numeric = df.select_dtypes(include=[np.number])
    predropped_shape = df_numeric.shape[0]
    #df_numeric = df_numeric.dropna()
    dropped_rows = predropped_shape - df_numeric.shape[0]

    # Calculate basic statistical measures using pandas describe method
    describe_df = df_numeric.describe().rename(index={
        'mean': 'Mean',
        'std': 'Standard_Deviation',
        '50%': 'Median',
        '25%': 'Q1',
        '75%': 'Q3',
        'min': 'Minimum',
        'max': 'Maximum',
    })

    # Extract precomputed statistics
    means = describe_df.loc['Mean']
    medians = describe_df.loc['Median']
    stds = describe_df.loc['Standard_Deviation']

    # Calculate additional statistical measures
    additional_stats = {
        'Count_nonNaN': df_numeric.count(),
        'Count_NaN': df_numeric.isna().sum(),
        'Variance': stds ** 2,
        'Skew': df_numeric.skew(),
        'Variance_coef%': stds / means * 100,
        'Mean_Median_Difference': means - medians,
        'Kurtosis': df_numeric.kurtosis(),
        'Median_Absolute_Deviation': df_numeric.apply(lambda x: median_abs_deviation(x, nan_policy='omit')),
        'Standard_Error_Mean': df_numeric.sem(),
        'Mean_Trimmed': df_numeric.apply(lambda x: trim_mean(x, 0.1)),
        'Standard_Deviation_relative%': stds / means * 100,
        'Outliers_Tukey': df_numeric.apply(count_outliers_tukey, axis=0),
        'Outliers_1p96_SD': df_numeric.apply(lambda x: count_outliers_1p96_sd(x, means[x.name], stds[x.name]), axis=0),
        'Mode': df_numeric.mode().iloc[0] if not df_numeric.mode().empty else np.nan, #todo giving series of NP.nan
        'Skew_coef': 3 * (means - medians) / stds,
        'Mean_Geometric': df_numeric.apply(lambda x: gmean(x[x > 0]) if (x > 0).any() else np.nan),
        'Mean_Harmonic': df_numeric.apply(lambda x: hmean(x[x > 0]) if (x > 0).any() else np.nan),
        'Kurtosis_Excess': df_numeric.kurtosis() - 3,
    }
    # for key, value in additional_stats.items():
    #     if hasattr(value, 'shape'):
    #         print(f"{key}: shape = {value.shape}")
    #     elif hasattr(value, 'size'):
    #         print(f"{key}: size = {value.size}")
    #     else:
    #         print(f"{key}: Not an array or Series")

    # Create DataFrame for additional statistics
    additional_stats_df = pd.DataFrame(additional_stats)

    # Merge and transpose the final DataFrame
    final_df = pd.concat([describe_df.T, additional_stats_df], axis=1).T

    # Reorder columns for final output
    ordered_columns = ['Count_nonNaN', 'Count_NaN',
        'Mean', 'Mean_Trimmed', 'Mean_Harmonic', 'Mean_Geometric', 'Median', 'Mode',
        'Standard_Deviation', 'Standard_Deviation_relative%', 'Variance', 'Variance_coef%',
        'Median_Absolute_Deviation', 'Standard_Error_Mean', 'Q1', 'Q3', 'Skew', 'Skew_coef',
        'Mean_Median_Difference', 'Kurtosis', 'Kurtosis_Excess',
        'Minimum', 'Maximum', 'Outliers_Tukey', 'Outliers_1p96_SD'
    ]

    final_df = final_df.reindex(ordered_columns).T

    if dropped_rows > 0:
        print(f"Warning: {dropped_rows} rows were dropped due to NaN values. If this is a relatively large amount of data lost this may affect statistical results.")

    final_df = final_df.apply(pd.to_numeric, errors = 'coerce')
    return final_df

# Example usage:
# from super_describe import super_describe
# df = pd.read_csv("path_to_your_file.csv")
# print(super_describe(df))