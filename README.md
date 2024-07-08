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