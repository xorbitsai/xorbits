import xorbits
from xorbits import dataframe as xd


if __name__ == '__main__':
    xorbits.new_session(
        address="test://127.0.0.1",
        backend="mars",
        init_local=True,
        default=True)
    df = xd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'a', 'c']})
    print(df.groupby('col2').max().execute())
