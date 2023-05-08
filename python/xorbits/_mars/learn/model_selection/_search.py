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


from numbers import Number
from typing import Dict, Iterable, List, Mapping, Sequence, Union

import numpy as np
from sklearn.model_selection._search import ParameterGrid as _SK_ParameterGrid

from ... import tensor as mt


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
                if isinstance(value, (np.ndarray, Number)):
                    xnp_value = mt.array(value)
                    value = xnp_value
                    grid[key] = xnp_value
                elif isinstance(value, Iterable):
                    is_num = True
                    for i, v in enumerate(value):
                        if isinstance(v, (np.ndarray, Number)):
                            xnp_value = mt.array(v)
                            value[i] = xnp_value
                            grid[key][i] = xnp_value
                        else:
                            is_num = False
                    if is_num:
                        grid[key] = mt.ExecutableTuple(grid[key])

                if isinstance(value, (mt.Tensor)) and value.ndim > 1:
                    raise ValueError(
                        f"Parameter array for {key!r} should be one-dimensional, got:"
                        f" {value!r} with shape {value.shape}"
                    )
                if isinstance(value, str) or not isinstance(
                    value, (mt.Tensor, Sequence)
                ):
                    raise TypeError(
                        f"Parameter grid for parameter {key!r} needs to be a list or a"
                        f" mars array, but got {value!r} (of type "
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
