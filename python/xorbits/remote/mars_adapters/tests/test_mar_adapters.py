# Copyright 2022 XProbe Inc.
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

import doctest
from doctest import DocTestFinder, DocTestRunner, TestResults
from typing import List

import pandas as pd
import pytest

from xorbits.core import DataRef

from .... import pandas as xpd
from .... import remote


def test_spawn(setup):
    def to_str(df):
        return str(df)

    # scaler arguments
    ret = remote.spawn(to_str, (1,))
    assert isinstance(ret, DataRef)
    import re

    assert re.match("Object <op=RemoteFunction, key=.*>", str(ret))
    assert ret.fetch() == "1"

    df = pd.DataFrame((1, 2, 3))
    xdf = xpd.DataFrame(df)
    ret = remote.spawn(to_str, (xdf,))
    assert re.match("Object <op=RemoteFunction, key=.*>", str(ret))
    assert ret.fetch() == "   0\n0  1\n1  2\n2  3"


def run_docstring(
    f,
    globs,
    verbose=False,
    name="NoName",
    compileflags=None,
    optionflags=0,
) -> List[TestResults]:
    finder = DocTestFinder(verbose=verbose, recurse=False)
    runner = DocTestRunner(verbose=verbose, optionflags=optionflags)

    results: List[TestResults] = []
    for test in finder.find(f, name, globs=globs):
        results.append(runner.run(test, compileflags=compileflags))

    return results


def test_spawn_examples(setup):
    results = run_docstring(
        remote.spawn,
        {},
        name="spawn",
        verbose=True,
        optionflags=doctest.NORMALIZE_WHITESPACE,
    )

    for result in results:
        if result.failed != 0:
            pytest.fail(f"{result.failed} out of {result.attempted} example(s) failed.")
        else:
            print(f"{result.attempted} example(s) passed.")
