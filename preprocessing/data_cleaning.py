"""
Clean Creditreform Dataset.

Following steps are undertaken.
    Select only observations between 1997-2002.
    Select only observations from the main five industries.
    remove observations with 0 values in variables used for denominator.
    Censor outliers per column by the 0.05- or 0.95-percentile respectively.
"""

import numpy as np
import pandas as pd


pd.read_csv()
