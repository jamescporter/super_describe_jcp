import pandas as pd
import numpy as np
from scipy.stats import chi2  # , mode, hmean, gmean, trim_mean, median_abs_deviation,
import time
import warnings


def super_describe(df, cast_to_float32 = False, verbose = False):
    """
    Generate a comprehensive set of statistical measures for each numeric column in a DataFrame.

    Parameters:
    df (pd.DataFrame): Input pandas DataFrame.
    cast_to_float32 (bool): If True, casts data to float32 to speed up calculation on modern CPUs.

    Returns:
    pd.DataFrame: DataFrame with statistical measures.
    """
    start_time = time.time()
    last_time = start_time  # tracks time of previous log

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    def print_w_time(msg: str):
        nonlocal last_time
        if verbose:
            now = time.time()
            elapsed = now - last_time  # time since previous call
            last_time = now  # update for next call
            print(f"Previous {elapsed:.3f}s - Next: {msg}")

    # Select only numeric columns from the DataFrame and drop rows with NaNs
    df_numeric = df.select_dtypes(include = [np.number])

    # Downcast to float32 for SIMD speedup if requested
    if cast_to_float32: df_numeric = df_numeric.astype('float32')

    predropped_shape = df_numeric.shape[0]
    # df_numeric = df_numeric.dropna()
    dropped_rows = predropped_shape - df_numeric.shape[0]
    print_w_time("Calculating basic statistics with describe()...")
    describe_df = df_numeric.describe(percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).rename(index = {
        'count':'Count_nonNaN',
        'mean': 'Mean',
        'std':  'Standard_Deviation',
        '50%':  'Median',
        '1%':   'P1',
        '5%':   'P5',
        '25%':  'Q1',
        '75%':  'Q3',
        '95%':  'P95',
        '99%':  'P99',
        'min':  'Minimum',
        'max':  'Maximum',
    })

    # Extract precomputed statistics
    print_w_time("Extracting precomputed statistics...")
    counts = describe_df.loc['Count_nonNaN']
    means = describe_df.loc['Mean']
    medians = describe_df.loc['Median']
    stds = describe_df.loc['Standard_Deviation']
    q1 = describe_df.loc['Q1']
    q3 = describe_df.loc['Q3']

    print_w_time("Calculating skewness and kurtosis...")
    skew_vals = df_numeric.skew()
    kurt_vals = df_numeric.kurtosis()  # Pandas calculates "Excess Kurtosis" (Normal = 0)

    print_w_time("Calculating outliers using Tukey...")
    iqr = q3 - q1
    lower_fence_tukey = q1 - 1.5 * iqr
    upper_fence_tukey = q3 + 1.5 * iqr
    outliers_tukey_counts = ((df_numeric < lower_fence_tukey) | (df_numeric > upper_fence_tukey)).sum()

    print_w_time("Calculating outliers using 1.96 SD method...")
    lower_fence_sd = means - 1.96 * stds
    upper_fence_sd = means + 1.96 * stds
    outliers_sd_counts = ((df_numeric < lower_fence_sd) | (df_numeric > upper_fence_sd)).sum()

    print_w_time("Calculating Zero-Inflation and Negatives counts...")
    zeros = (df_numeric == 0).sum()
    negatives = (df_numeric < 0).sum()

    print_w_time("Calculating Jarque-Bera statistics...")
    # Formula: (n/6) * (S^2 + (K^2)/4) using Excess Kurtosis
    jb_score = (counts / 6) * (skew_vals ** 2 + (kurt_vals ** 2) / 4)

    print_w_time("Calculating Jarque-Bera p-values...") #p-value from Chi-Squared distribution (df=2)
    jb_p_value = pd.Series(chi2.sf(jb_score, 2), index = df_numeric.columns)

    print_w_time("Calculating mode...")
    # Try Scipy mode first (Fast C-code). Fallback to Pandas if mixed-types fail.
    try:
        # nan_policy='omit' handles NaNs; getting [0] extracts the mode array from the result object
        mode_result = mode(df_numeric.values, axis = 0, nan_policy = 'omit')
        # If all values are NaN, mode is empty, causing IndexError. In that case, fallback.
        if mode_result.mode.size == 0: raise ValueError("Empty mode result")
        mode_vals = pd.Series(mode_result.mode.flatten(), index = df_numeric.columns)
    except Exception:
        mode_vals = df_numeric.mode(dropna = True).iloc[0]
        mode_vals = mode_vals.reindex(df_numeric.columns)

    print_w_time("Compiling final statistics DF & additional stats...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        additional_stats = {
            'Count_nonNaN':             counts,
            'Count_NaN':                predropped_shape - counts,
            'Sum':                      df_numeric.sum(),
            'Variance':                 stds ** 2,
            'Skew':                     skew_vals,
            'Variance_coef%':           stds / means * 100,
            'Mean_Median_Difference':   means - medians,
            'Kurtosis':                 kurt_vals,
            'Mean_Absolute_Deviation':  (df_numeric - means).abs().mean(),
            'Median_Absolute_Deviation':(df_numeric - medians).abs().median(),
            'Standard_Error_Mean':      stds / np.sqrt(counts),
            # 'Mean_Trimmed': df_numeric.apply(lambda x: trim_mean(x, 0.1)),
            # 'Standard_Deviation_relative%': stds / means * 100,
            'Outliers_Tukey':           outliers_tukey_counts,
            'Outliers_1p96_SD':         outliers_sd_counts,
            'Mode':                     mode_vals,
            'Skew_coef':                3 * (means - medians) / stds,
            'IQR':                      iqr,
            'Range':                    describe_df.loc['Maximum'] - describe_df.loc['Minimum'],
            'Count_Zeros':              zeros,
            'Percent_Zeros':            (zeros / counts) * 100,
            'Percent_Negative':         (negatives / counts) * 100,
            'Count_Unique':             df_numeric.nunique(),
            'Unique_Ratio':             df_numeric.nunique() / counts,
            'Jarque_Bera_Prob':         jb_p_value,

            # 'Mean_Geometric': df_numeric.apply(lambda x: gmean(x[x > 0]) if (x > 0).any() else np.nan),
            # 'Mean_Harmonic': df_numeric.apply(lambda x: hmean(x[x > 0]) if (x > 0).any() else np.nan),
            # 'Kurtosis_Excess': df_numeric.kurtosis() - 3,
        }

    print_w_time("Creating additional statistics DataFrame...")
    additional_stats_df = pd.DataFrame(additional_stats)

    # Merge and transpose the final DataFrame
    final_df = pd.concat([describe_df.T, additional_stats_df], axis = 1)

    # Remove duplicate index (e.g. Count_nonNaN appearing in both)
    final_df = final_df.loc[~final_df.index.duplicated(keep = 'first')]

    print_w_time("Reordering columns for final output...")
    ordered_columns = [
        'Count_nonNaN', 'Count_NaN', 'Sum',
        'Mean',  # 'Mean_Trimmed',
        # 'Mean_Harmonic', 'Mean_Geometric',
        'Median', 'Mode',
        'Standard_Deviation',  # 'Standard_Deviation_relative%',
        'Variance', 'Variance_coef%',
        'Mean_Absolute_Deviation',
        'Median_Absolute_Deviation', 'Standard_Error_Mean',
        'Minimum', 'P1', 'P5', 'Q1', 'Q3', 'P95', 'P99', 'Maximum',  # Ordered Percentiles
        'Range', 'IQR',
        'Skew', 'Skew_coef',
        'Mean_Median_Difference', 'Kurtosis', 'Jarque_Bera_Prob',  # 'Kurtosis_Excess',
        'Count_Zeros', 'Percent_Zeros', 'Percent_Negative', 'Count_Unique', 'Unique_Ratio',
        'Outliers_Tukey', 'Outliers_1p96_SD'
    ]

    final_df = final_df.reindex(ordered_columns).T

    if dropped_rows > 0:
        print(f"Warning: {dropped_rows} rows were dropped due to NaN values. If this is a relatively large amount of data lost this may affect statistical results.")

    print_w_time("Converting final DataFrame to numeric types...")
    final_df = final_df.apply(pd.to_numeric, errors = 'coerce')
    print_w_time("...super_describe completed.\n")
    return final_df

# Example usage:
# from super_describe import super_describe
# df = pd.read_csv("path_to_your_file.csv")
# print(super_describe(df, cast_to_float32=True))