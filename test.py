import pandas as pd
import numpy as np
import random
import python.xorbits._mars.dataframe as md
from python.xorbits._mars.core import tile
import pytest

# import xorbits
# import xorbits.pandas as md

df = pd.DataFrame([[1, 2.12], [3.356, 4.567]])
res1 = df.applymap(lambda x: len(str(x)))
res2 = df.apply(lambda x: len(str(x)))
print(res1)
print()
print(res2)
exit()

df_raw = pd.DataFrame(
    [
        [np.nan, 2, np.nan, 0],
        [3, 4, np.nan, 1],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, 3, np.nan, 4],
    ],
    columns=list("ABCD"),
)

df = md.DataFrame(df_raw, chunk_size=2)

# test DataFrame single chunk with numeric fill
r = df.fillna(method="ffill")
tile(r)
r.execute().fetch()
# pd.testing.assert_frame_equal(r.execute().fetch(), df_raw.fillna(1))

# # test DataFrame single chunk with value as single chunk
# value_df = md.DataFrame(value_df_raw)
# r = df.fillna(value_df)
