A function that takes a pandas DataFrame as an input and returns a transposed DataFrame with the calculated mean, median, standard deviation,
variation, skewness, coefficient of variation as percentage, mean/median difference, and kurtosis for each numeric column.

Exists as a single python file which can easily be imported as a funtion within your IDE.


## Installation

Place `super_describe.py` in your project directory.

## Usage

```python
import pandas as pd
from super_describe import super_describe

# Load your data into a pandas DataFrame
df = pd.read_csv("path_to_your_file.csv")

# Generate statistical description
result = super_describe(df)
print(result)
```

## Statistical Measures Calculated by `super_describe`

To enhance the table with additional columns for robustness and interpretation, here’s how you can modify it:

| Statistical Measure                | Description                                         | Units          | Comparable across cols?  | Robustness                     | More robust than   |
|------------------------------------|-----------------------------------------------------|----------------|--------------------------|--------------------------------|--------------------|
| Mean                               | Average                                             | Data           | No                       | No - outliers                  | -                  |
| Mean Trimmed                       | Average after trimming extremes                     | Data           | No                       | Yes - outliers                 | Mean               |
| Mean Harmonic                      | Harmonic mean                                       | Data           | No                       | No - extreme values            | -                  |
| Mean Geometric                     | Geometric mean                                      | Data           | No                       | Yes - extreme values           | -                  |
| Median                             | Middle value                                        | Data           | No                       | Yes - outliers                 | Mean, Mean trimmed |
| Mode                               | Most frequent value                                 | Data           | No                       | Yes - outliers                 | Mean, Median       |
| Standard Deviation (STDDEV)        | Dispersion of values relative to mean               | Data           | No                       | No - outliers                  | MAD, IQR           |
| STDDEV Relative % (RSD)            | STDDEV as (% of mean)                               | %              | Yes                      | -                              | -                  |
| Variance                           | STDDEV^2                                            | Data^2         | No                       | No - outliers                  | -                  |
| Variance Coefficient %             | STDDEV / mean (as % or ratio) - Similar to RSD      | %              | Yes                      | -                              | -                  |
| Median Absolute Deviation (MAD)    | Variability measure based on median (more robust)   | Data           | No                       | Yes - outliers                 | STDDEV, Variance   |
| Standard Error of Mean (SEM)       | STDDEV of sample-mean estimate                      | Data           | No                       | -                              | -                  |
| Q1 (First Quartile)                | Median of lower half                                | Data           | No                       | Yes - outliers                 | -                  |
| Q3 (Third Quartile)                | Median of upper half                                | Data           | No                       | Yes - outliers                 | -                  |
| Skew                               | Asymmetry measure                                   | Dimensionless  | Yes                      | Partly - extreme values        | -                  |
| Skew Coefficient                   | Degree of asymmetry of distribution around its mean | Dimensionless  | Yes                      | -                              | -                  |
| Mean-Median Difference             | Difference between mean and median                  | Data           | No                       | No - outliers                  | -                  |
| Kurtosis                           | "Tailedness" measure                                | Dimensionless  | Yes                      | Partly - extreme values        | -                  |
| Kurtosis Excess                    | "Peakedness" compared to normal distribution        | Dimensionless  | Yes                      | -                              | -                  |
| Minimum                            | Smallest value                                      | Data           | No                       | Yes - outliers                 | -                  |
| Maximum                            | Largest value                                       | Data           | No                       | Yes - outliers                 | -                  |
| Outliers (Tukey Method)            | Outliers count using Tukey's method (1.5 * IQR)     | Count          | No                       | Yes - skewed distributions     | -                  |
| Outliers (1.96 Standard Deviation) | Outliers count using 1.96 * Std. Dev rule.          | Count          | No                       | Yes -  symmetric distributions | -                  |