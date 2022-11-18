# -*- coding: utf-8 -*-
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
import inspect
from doctest import DocTestFinder, DocTestRunner, TestResults
from typing import List

import pytest

from .... import numpy as xnp


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
        # exclude examples that plot graphs.
        examples = []
        for example in test.examples:
            if example.source.startswith("plt.") and not example.want:
                continue
            examples.append(example)
        test.examples = examples

        results.append(runner.run(test, compileflags=compileflags))

    return results


parameters = []
parameters.extend(
    [
        (xnp, name)
        for name, _ in inspect.getmembers(xnp, inspect.isfunction)
        if not name.startswith("_")
    ]
)
parameters.extend(
    [
        (xnp.fft, name)
        for name, _ in inspect.getmembers(xnp.fft, inspect.isfunction)
        if not name.startswith("_")
    ]
)
parameters.extend(
    [
        (xnp.random, name)
        for name, _ in inspect.getmembers(xnp.random, inspect.isfunction)
        if not name.startswith("_")
    ]
)
parameters.extend(
    [
        (xnp.linalg, name)
        for name, _ in inspect.getmembers(xnp.linalg, inspect.isfunction)
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
