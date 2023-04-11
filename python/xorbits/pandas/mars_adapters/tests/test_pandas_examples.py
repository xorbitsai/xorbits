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

import doctest
import inspect
from doctest import DocTestFinder, DocTestRunner, TestResults
from typing import Any, List, Tuple

import pytest

from .... import pandas as xpd


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


parameters: List[Tuple[Any, Any]] = []
parameters.extend(
    [
        (xpd, name)
        for name, _ in inspect.getmembers(xpd, inspect.isfunction)
        if not name.startswith("_")
    ]
)
xdf = xpd.DataFrame({"foo": (1,)})
parameters.extend(
    [
        (xdf, name)
        for name, _ in inspect.getmembers(xdf, inspect.isfunction)
        if not name.startswith("_")
    ]
)
xs = xpd.Series((1,))
parameters.extend(
    [
        (xs, name)
        for name, _ in inspect.getmembers(xs, inspect.isfunction)
        if not name.startswith("_")
    ]
)


@pytest.mark.parametrize("obj,name", parameters)
def test_docstrings(setup, doctest_namespace, obj, name):
    results = run_docstring(
        getattr(obj, name),
        doctest_namespace,
        name=name,
        verbose=True,
        optionflags=doctest.NORMALIZE_WHITESPACE,
    )

    for result in results:
        if result.failed != 0:
            pytest.fail(f"{result.failed} out of {result.attempted} example(s) failed.")
        else:
            print(f"{result.attempted} example(s) passed.")
