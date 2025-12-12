import pandas as pd
import numpy as np
from scipy.stats import chi2, mode#, hmean, gmean, trim_mean, median_abs_deviation,
import time
import warnings
import numba


@numba.jit(nopython = True, parallel = True)
def fast_stats_numba(vals):
    n_rows, n_cols = vals.shape
    modes = np.full(n_cols, np.nan)
    unique_counts = np.zeros(n_cols, dtype = np.int64)

    for i in numba.prange(n_cols):
        col = vals[:, i].copy()
        col.sort()

        # Mode variables
        max_freq = 0
        current_freq = 0
        current_val = np.nan
        best_mode = np.nan

        # Unique variable
        u_count = 0

        for j in range(n_rows):
            val = col[j]
            if np.isnan(val): continue

            # Check for new unique value
            if np.isnan(current_val) or val != current_val:
                u_count += 1

                # Mode logic (Process previous run)
                if current_freq > max_freq:
                    max_freq = current_freq
                    best_mode = current_val

                current_val = val
                current_freq = 1
            else:
                current_freq += 1

        # Check final run
        if current_freq > max_freq:
            best_mode = current_val

        modes[i] = best_mode
        unique_counts[i] = u_count

    return modes, unique_counts

def super_describe(df, cast_to_float32 = False, verbose = False):
    """
    Generate a comprehensive set of statistical measures for each numeric column in a DataFrame.

    Parameters:
    df (pd.DataFrame): Input pandas DataFrame.
    cast_to_float32 (bool): If True, casts data to float32 to speed up calculation on modern CPUs.

    Returns:
    pd.DataFrame: DataFrame with statistical measures.
    """
    eps = 1e-12 # Small constant to prevent division by zero
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
            print(f"Previous interval {elapsed:.4f}s - Next: {msg}")

    # Select only numeric columns from the DataFrame and drop rows with NaNs
    df_numeric = df.select_dtypes(include = [np.number])

    # Downcast to float32 for SIMD speedup if requested
    if cast_to_float32: df_numeric = df_numeric.astype('float32')

    predropped_shape = df_numeric.shape[0]
    # df_numeric = df_numeric.dropna()
    dropped_rows = predropped_shape - df_numeric.shape[0]


    print_w_time("Calculating basic statistics with numpy vectorised...")
    #vals = df_numeric.values  # Work with raw array for speed
    # Precise: 'copy=False' attempts to use existing memory (view) if dtypes match.
    # It only creates a copy if the data is mixed, Nullable (Int64), or not float.
    # 1. Grab values as-is (Fastest, usually creates a view, no memory copy)
    #vals = df_numeric.to_numpy()
    vals = np.ascontiguousarray(df_numeric.values)

    # 2. Check for object fallback (caused by Nullable Ints, Mixed Types, etc.)
    if vals.dtype == 'object':
        if verbose:
            problem_cols = [
                col for col in df_numeric.columns
                if pd.api.types.is_extension_array_dtype(df_numeric[col].dtype)
                   or df_numeric[col].dtype == 'object'
            ]
            print(f"\n  -> Optimization Info: 'Object' array detected. Forcing float copy.")
            print(f"  -> Culprit columns (Nullable/Mixed): {problem_cols[:10]} {'...' if len(problem_cols) > 10 else ''}\n")

        # Use Pandas to convert. It correctly maps pd.NA -> np.nan
        vals = df_numeric.to_numpy(dtype = np.float32 if cast_to_float32 else float, na_value = np.nan)
    # ---------------------------------

    cols = df_numeric.columns
    n = np.sum(~np.isnan(vals), axis = 0) # Count valid values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        # Vectorised Central Tendency & Dispersion
        means_arr = np.nanmean(vals, axis = 0)
        stds_arr = np.nanstd(vals, axis = 0, ddof = 1)  # ddof=1 matches Pandas
        mins_arr = np.nanmin(vals, axis = 0)
        maxs_arr = np.nanmax(vals, axis = 0)

        # Vectorised Percentiles (Single pass sort)
        p_res = np.nanpercentile(vals, [1, 5, 25, 50, 75, 95, 99], axis = 0) # Indices: 0=1%, 1=5%, 2=25%, 3=50%, 4=75%, 5=95%, 6=99%
        median_arr = p_res[3]

    print_w_time("Calculating skewness, kurtosis AND MADs (Combined)...")
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        # OPTIMISATION: Calculate centered moments first (Multiplication is faster than Division)
        # 1. Center the data
        centered = vals - means_arr[None, :]

        # --- MAD CALCULATION (Reuse 'centered' to save time) ---
        # We need abs(centered) for MAD
        abs_centered = np.abs(centered)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = RuntimeWarning)
            mad_mean = np.nanmean(abs_centered, axis = 0)
            mad_median = np.nanmedian(np.abs(vals - median_arr[None, :]), axis = 0)
        # -------------------------------------------------------

        # 2. Calculate raw moments (Reuse memory where possible)
        centered_sq = centered * centered  # (x-u)^2
        centered_cu = centered_sq * centered  # (x-u)^3
        centered_qu = centered_sq * centered_sq  # (x-u)^4

        # 3. Sum moments (Collapse to 1D array)
        # 4. Normalise results at the end (Scalar division - instant)
        # Skew = Sum((x-u)^3) / std^3
        sum_z3 = np.nansum(centered_cu, axis = 0)/(stds_arr ** 3)
        sum_z4 = np.nansum(centered_qu, axis = 0)/(stds_arr ** 4)
        n_adj = n.astype(float)

        # Skewness Formula (Bias-corrected)
        skew_coef = n_adj / ((n_adj - 1) * (n_adj - 2))
        skew_vals = skew_coef * sum_z3
        skew_vals[n < 3] = np.nan

        # Kurtosis Formula (Excess, Bias-corrected)
        k1 = (n_adj * (n_adj + 1)) / ((n_adj - 1) * (n_adj - 2) * (n_adj - 3))
        k2 = (3 * (n_adj - 1) ** 2) / ((n_adj - 2) * (n_adj - 3))
        kurt_vals = (k1 * sum_z4) - k2
        kurt_vals[n < 4] = np.nan


    print_w_time("Calculating outliers using Tukey...")
    q1_arr, q3_arr = p_res[2], p_res[4]
    iqr_arr = q3_arr - q1_arr
    outliers_tukey = np.sum((vals < (q1_arr - 1.5 * iqr_arr)) | (vals > (q3_arr + 1.5 * iqr_arr)), axis = 0)


    print_w_time("Calculating outliers using 1.96 SD method...")
    outliers_sd = np.sum((vals < (means_arr - 1.96 * stds_arr)) | (vals > (means_arr + 1.96 * stds_arr)), axis=0)

    print_w_time("Calculating Zero-Inflation and Negatives counts...")
    # zeros = (df_numeric == 0).sum()
    # negatives = (df_numeric < 0).sum()
    zeros = np.sum(vals == 0, axis = 0)
    negatives = np.sum(vals < 0, axis = 0)

    #print_w_time("Calculating MADs...")
    # MAD Calculations (Huge speedup vs df - mean)
    # abs(vals - mean) broadcasted
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=RuntimeWarning)
    #     mad_mean = np.nanmean(np.abs(vals - means_arr[None, :]), axis=0)
    #     mad_median = np.nanmedian(np.abs(vals - median_arr[None, :]), axis=0)

    print_w_time("Preparing arrays for vectorized calculations...")
    # Vectorized Safe Division (Replaces .where)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        var_coef = np.where(np.abs(means_arr) > eps, (stds_arr / means_arr) * 100, np.nan)
        sem = np.where(n >= 2, stds_arr / np.sqrt(n), np.nan)
        skew_coef_val = np.where(stds_arr != 0, 3 * (means_arr - median_arr) / stds_arr, np.nan)
        #unique_ratio = df_numeric.nunique().values / n  # df.nunique() is hard to beat in pure numpy
        # NOTE: unique_ratio removed from here, moved to Mode block

    print_w_time("Calculating Jarque-Bera statistics...")
    # Formula: (n/6) * (S^2 + (K^2)/4) using Excess Kurtosis
    jb_score = (n / 6) * (skew_vals ** 2 + (kurt_vals ** 2) / 4)
    #jb_p_value = pd.Series(chi2.sf(jb_score, 2), index = df_numeric.columns) #p-value from Chi-Squared distribution (df=2)
    #jb_p_value = chi2.sf(jb_score, 2)
    jb_p_value = np.exp(-jb_score / 2) #exact solution for df=2


    print_w_time("Calculating Mode & Unique Counts (Numba Parallel)...")
    try:
        # ATTEMPT 1: Numba Parallel (Get Mode + Unique Counts in one go)
        mode_vals, unique_counts = fast_stats_numba(vals)
    except Exception:
        # FALLBACK: Numba failed (likely incompatible type or memory issue)
        # We must calculate unique counts the slow way now
        unique_counts = df_numeric.nunique().values

        try:
            # ATTEMPT 2: Scipy Mode (Fast C-code)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mode_result = mode(vals, axis = 0, nan_policy = 'omit')
            if mode_result.mode.size == 0: raise ValueError("Empty mode result")
            mode_vals = mode_result.mode.flatten()
        except Exception:
            # ATTEMPT 3: Pandas Mode (Slowest but most robust)
            mode_vals = df_numeric.mode(dropna = True).iloc[0].values

    # Calculate unique ratio (safe division)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        unique_ratio = unique_counts / n

    print_w_time("Compiling final statistics DF & additional stats...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category = RuntimeWarning)
        additional_stats = {
            'Count_nonNaN':             n,
            'Count_NaN':                predropped_shape - n,
            'Sum':                      np.nansum(vals, axis=0),
            'Mean':                     means_arr,
            'Median':                   median_arr,
            'Standard_Deviation':       stds_arr,
            'Variance':                 stds_arr ** 2,
            'Minimum':                  mins_arr,
            'Maximum':                  maxs_arr,
            'P1': p_res[0], 'P5': p_res[1], 'Q1': p_res[2], 'Q3': p_res[4], 'P95': p_res[5], 'P99': p_res[6],
            'Skew':                     skew_vals,
            'Variance_coef%':           var_coef,
            'Mean_Median_Difference':   means_arr - median_arr,
            'Kurtosis':                 kurt_vals,
            'Mean_Absolute_Deviation':  mad_mean,
            'Median_Absolute_Deviation':mad_median,
            'Standard_Error_Mean':      sem, #stds / np.sqrt(counts),
            # 'Mean_Trimmed': df_numeric.apply(lambda x: trim_mean(x, 0.1)),
            # 'Standard_Deviation_relative%': stds / means * 100,
            'Outliers_Tukey':           outliers_tukey,
            'Outliers_1p96_SD':         outliers_sd,
            'Mode':                     mode_vals,
            'Skew_coef':                skew_coef_val,
            'IQR':                      iqr_arr,
            'Range':                    maxs_arr - mins_arr,
            'Count_Zeros':              zeros,
            'Percent_Zeros':            (zeros / n) * 100, # n is already float array or int, safe
            'Percent_Negative':         (negatives / n) * 100,
            'Count_Unique':             df_numeric.nunique().values,
            'Unique_Ratio':             unique_ratio,
            'Jarque_Bera_Prob':         jb_p_value,

            # 'Mean_Geometric': df_numeric.apply(lambda x: gmean(x[x > 0]) if (x > 0).any() else np.nan),
            # 'Mean_Harmonic': df_numeric.apply(lambda x: hmean(x[x > 0]) if (x > 0).any() else np.nan),
            # 'Kurtosis_Excess': df_numeric.kurtosis() - 3,
        }
    # Construct DataFrame once from dict of arrays (Fastest method)
    final_df = pd.DataFrame(additional_stats, index = cols)

    print_w_time("Reordering columns for final output & converting final df...")
    ordered_columns = [
        'Count_nonNaN', 'Count_NaN', 'Sum',
        'Mean',  # 'Mean_Trimmed',
        # 'Mean_Harmonic', 'Mean_Geometric',
        'Median', 'Mode',
        'Standard_Deviation',  # 'Standard_Deviation_relative%',
        'Variance', 'Variance_coef%',
        'Mean_Absolute_Deviation', 'Median_Absolute_Deviation', 'Standard_Error_Mean',
        'Minimum', 'P1', 'P5', 'Q1', 'Q3', 'P95', 'P99', 'Maximum',  # Ordered Percentiles
        'Range', 'IQR',
        'Skew', 'Skew_coef',
        'Mean_Median_Difference', 'Kurtosis', 'Jarque_Bera_Prob',  # 'Kurtosis_Excess',
        'Count_Zeros', 'Percent_Zeros', 'Percent_Negative', 'Count_Unique', 'Unique_Ratio',
        'Outliers_Tukey', 'Outliers_1p96_SD'
    ]
    #final_df = final_df.reindex(ordered_columns).T
    final_df = final_df.reindex(columns = ordered_columns).T

    if dropped_rows > 0:
        print(f"Warning: {dropped_rows} rows were dropped due to NaN values. If this is a relatively large amount of data lost this may affect statistical results.")
    #final_df = final_df.apply(pd.to_numeric, errors = 'coerce')
    print_w_time("...super_describe completed.")
    if verbose: print("\n")
    return final_df

# Example usage:
# from super_describe import super_describe
# df = pd.read_csv("path_to_your_file.csv")
# print(super_describe(df, cast_to_float32=True))