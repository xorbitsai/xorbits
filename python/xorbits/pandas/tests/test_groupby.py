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

import pandas as pd

from ... import pandas as xpd


def test_groupby_rolling(setup):
    df = pd.DataFrame({"Group": ["A", "A", "B", "B", "B"], "Value": [1, 2, 3, 4, 5]})
    xdf = xpd.DataFrame(df)

    rolling_x_mean = xdf.groupby("Group")["Value"].rolling(window=2).mean().to_pandas()
    rolling_mean = df.groupby("Group")["Value"].rolling(window=2).mean()

    pd.testing.assert_series_equal(rolling_x_mean, rolling_mean)
