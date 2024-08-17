# Tests for wheel
import warnings

import numpy
import pandas
import requests


def test_basic_cases():
    with warnings.catch_warnings():        
        # "error" help us find the deprecated APIs
        warnings.simplefilter("error")
        import xorbits
        import xorbits.pandas as pd
        import xorbits.numpy as np

    xorbits.init()
    df = pd.DataFrame({'A': [1, 2, 3]})
    assert str(df) == str(pandas.DataFrame({'A': [1, 2, 3]}))

    array = np.ones((2, 2))
    assert str(array) == str(numpy.ones((2, 2)))

    # make sure web is built
    from xorbits._mars.deploy.oscar.session import get_default_session

    default_session = get_default_session()
    web_url = default_session.get_web_endpoint()
    js_url = f'{web_url}/static/bundle.js'
    assert requests.get(js_url).status_code == 200
    default_session.stop_server()
