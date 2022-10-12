import _mars.dataframe as md


def test_mars_functionalities():
    df = md.DataFrame({'a': [0, 1, 2]})
    assert df.sum().shape == (1,)
