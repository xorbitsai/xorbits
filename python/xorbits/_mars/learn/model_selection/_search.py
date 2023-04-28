# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
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


from typing import Dict, Iterable, List, Mapping, Sequence, Union

import numpy as np
from sklearn.model_selection._search import BaseSearchCV as _SK_BaseSearchCV
from sklearn.model_selection._search import GridSearchCV as _SK_GridSearchCV
from sklearn.model_selection._search import ParameterGrid as _SK_ParameterGrid

from ... import tensor as mt

# from ..utils.validation import indexable


class ParameterGrid(_SK_ParameterGrid):
    param_grid: List[Dict[str, Union[mt.Tensor, Sequence]]]

    def __init__(
        self, param_grid: Union[List[Dict[str, Iterable]], Mapping[str, Iterable]]
    ):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError(
                f"Parameter grid should be a dict or a list, got: {param_grid!r} of"
                f" type {type(param_grid).__name__}"
            )

        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]

        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError(f"Parameter grid is not a dict ({grid!r})")
            for key, value in grid.items():
                if isinstance(value, np.ndarray):
                    xnp_value = mt.array(value)
                    value = xnp_value
                    grid[key] = xnp_value

                if isinstance(value, (mt.Tensor, Sequence)) and value.ndim > 1:
                    raise ValueError(
                        f"Parameter array for {key!r} should be one-dimensional, got:"
                        f" {value!r} with shape {value.shape}"
                    )
                if isinstance(value, str) or not isinstance(
                    value, (mt.Tensor, Sequence)
                ):
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} needs to be a list or a"
                        f" numpy array, but got {value!r} (of type "
                        f"{type(value).__name__}) instead. Single values "
                        "need to be wrapped in a list with one element."
                    )
                if len(value) == 0:
                    raise ValueError(
                        f"Parameter grid for parameter {key!r} need "
                        f"to be a non-empty sequence, got: {value!r}"
                    )

        self.param_grid = param_grid

    def __getitem__(self, ind):
        out = super().__getitem__(ind)
        for k, v in out.items():
            if isinstance(v, mt.Tensor):
                out[k] = v.execute()
        return out


class BaseSearchCV(_SK_BaseSearchCV):
    def fit(self, X, y=None, *, groups=None, **fit_params):
        _dtype = [mt.float64, mt.float32]
        X, y = self._validate_data(X, y, accept_sparse=True, dtype=_dtype, order="C")
        # estimator = self.estimator
        # refit_metric = "score"

        # # only support using estimator fit for now
        # scorers = self.estimator.score

        # X, y, groups = indexable(X, y, groups)
        # fit_params = _check_fit_params(X, fit_params)

        # cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        # n_splits = cv_orig.get_n_splits(X, y, groups)

        # base_estimator = clone(self.estimator)

        # fit_and_score_kwargs = dict(
        #     scorer=scorers,
        #     fit_params=fit_params,
        #     return_train_score=self.return_train_score,
        #     return_n_test_samples=True,
        #     return_times=True,
        #     return_parameters=False,
        #     error_score=self.error_score,
        #     verbose=self.verbose,
        # )
        # results = {}


class GridSearchCV(_SK_GridSearchCV):
    pass
