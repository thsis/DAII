"""
Define Finantial Ratios.abs

Following steps are undertaken:
    - Read cleaned data.
    - Define finantial ratios.
    - Censor values for ratios if they fall outside of
      [0.05-percentile, 0.95-percentile] by the respective value.
"""

import pandas as pd
import numpy as np
import os

data_path = os.path.join("data", "credit_clean.csv")
data = pd.read_csv(data_path, sep=';')
data.head()
data.shape

# Define ratios.
x1 = data.VAR22 / data.VAR6,
x2 = data.VAR22 / data.VAR16,
x3 = data.VAR21 / data.VAR6,
x4 = data.VAR21 / data.VAR16,
x5 = data.VAR20 / data.VAR6,
x6 = (data.VAR20 + data.VAR18) / data.VAR6,
x7 = data.VAR20 / data.VAR16,
x8 = data.VAR9 / data.VAR6,
x9 = (data.VAR9 - data.VAR5) / (data.VAR6-data.VAR5-data.VAR1-data.VAR8),
x10 = data.VAR12 / data.VAR6,
x11 = (data.VAR12 - data.VAR1) / data.VAR6,
x12 = (data.VAR12 + data.VAR13) / data.VAR6,
x13 = data.VAR14 / data.VAR6,
x14 = data.VAR20 / data.VAR19,
x15 = data.VAR1 / data.VAR6,
x16 = data.VAR1 / data.VAR12,
x17 = (data.VAR3 - data.VAR2) / data.VAR12,
x18 = data.VAR3 / data.VAR12,
x19 = (data.VAR3 - data.VAR12) / data.VAR6,
x20 = data.VAR12 / (data.VAR12 + data.VAR13),
x21 = data.VAR6 / data.VAR16,
x22 = data.VAR2 / data.VAR16,
x23 = data.VAR7 / data.VAR16,
x24 = data.VAR15 / data.VAR16,
x25 = np.log(data.VAR6),
x26 = data.VAR23 / data.VAR2,
x27 = data.VAR24 / (data.VAR12 + data.VAR13),
x28 = data.VAR25 / data.VAR1

ratios = pd.DataFrame(data={"ID": data.ID,
                            "x1": x1})
