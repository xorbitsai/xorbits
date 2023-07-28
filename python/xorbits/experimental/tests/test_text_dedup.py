# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from ... import pandas as pd
from ...experimental import dedup


def test_dedup_execute(setup):
    words = list("abcdefghijklmnopqrstuvwxyz")
    df = pd.DataFrame(
        {
            "text": [
                " ".join(["".join(np.random.choice(words, 5)) for _ in range(50)])
                for _ in np.arange(9)
            ]
            * 2
            + [
                " ".join(["".join(np.random.choice(words, 4)) for _ in range(50)])
                for _ in np.arange(2)
            ],
        }
    )

    # test exact one chunk
    result = dedup(df, col="text", method="exact").execute().fetch()
    assert result.shape[0] == 11

    # test exact multi chunks
    result = dedup(df.rechunk(1), col="text", method="exact").execute().fetch()
    assert result.shape[0] == 11

    # test minhash one chunk
    result = dedup(df, col="text", method="minhash").execute().fetch()
    assert result.shape[0] == 11

    # test minhash multi chunks
    result = dedup(df.rechunk(1), col="text", method="minhash").execute().fetch()
    assert result.shape[0] == 11

    # test error threshold
    with pytest.raises(ValueError):
        dedup(df, col="text", threshold="0.7").execute().fetch()

    with pytest.raises(ValueError):
        dedup(df, col="text", threshold=2).execute().fetch()

    # test error num_perm
    with pytest.raises(ValueError):
        dedup(df, col="text", num_perm=1.5).execute().fetch()

    # test error min_length
    with pytest.raises(ValueError):
        dedup(df, col="text", min_length=1.5).execute().fetch()

    # test error ngrams
    with pytest.raises(ValueError):
        dedup(df, col="text", ngrams=1.5).execute().fetch()

    # test error seed
    with pytest.raises(ValueError):
        dedup(df, col="text", seed=1.5).execute().fetch()

    # test error text column
    with pytest.raises(ValueError):
        dedup(df, col="non-exist").execute().fetch()

    # test error method
    with pytest.raises(ValueError):
        dedup(df, col="text", method="non-exist").execute().fetch()
