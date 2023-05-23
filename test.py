from xorbits._mars.dataframe.datasource.dataframe import from_pandas
import pandas as pd

df = pd.DataFrame({
        "lev1": [1, 1, 1, 2, 2, 2],
        "lev2": [1, 1, 2, 1, 1, 2],
        "lev3": [1, 2, 1, 2, 1, 2],
        "lev4": [1, 2, 3, 4, 5, 6],
        "values": [0, 1, 2, 3, 4, 5]})

xdf = from_pandas(df, chunk_size=3)
print(df.pivot(index="lev1", columns=["lev2", "lev3"], values="values"))
print(xdf.pivot(index="lev1", columns=["lev2", "lev3"], values="values").execute().fetch().sort_index(axis=0).sort_index(axis=1).astype(float))