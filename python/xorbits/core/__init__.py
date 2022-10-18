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
import traceback
from typing import Any, Callable

from .data import Proxy, XorbitsData, XorbitsDataRef, to_mars, wrap_mars_callable


def _install_magic_methods():
    def wrap_setattr(pass_through: bool = True) -> Callable[[Any, str, Any], None]:
        def wrapped(self: Any, name: str, value: Any) -> None:
            if name.startswith("_"):
                object.__setattr__(self, name, value)
            else:
                if pass_through:
                    getattr(self.proxied, "__setattr__")(name, value)
                else:
                    getattr(type(self.proxied), name).fset(self.proxied, to_mars(value))

        return wrapped

    def wrap_magic_method(
        method_name: str, pass_through: bool = True
    ) -> Callable[[Any], Any]:
        """
        TODO: docstring
        """

        def wrapped(*args, **kwargs):
            assert len(args) >= 1
            assert isinstance(args[0], Proxy)
            self: "Proxy" = args[0]

            if hasattr(self.proxied, method_name):
                if pass_through:
                    return getattr(self.proxied, method_name)(*args[1:], **kwargs)
                else:
                    return wrap_mars_callable(getattr(self.proxied, method_name))(
                        *args[1:], **kwargs
                    )
            else:
                raise AttributeError(
                    f"'{type(self)}' object has no attribute '{method_name}'"
                )

        return wrapped

    _magic_method_names = [
        "__abs__",
        "__add__",
        "__and__",
        "__div__",
        "__eq__",
        "__floordiv__",
        "__ge__",
        "__getitem__",
        "__gt__",
        "__invert__",
        "__le__",
        "__len__",
        "__lt__",
        "__matmul__",
        "__mod__",
        "__mul__",
        "__ne__",
        "__neg__",
        "__or__",
        "__pow__",
        "__radd__",
        "__rand__",
        "__rdiv__",
        "__reduce__",
        "__reduce_ex__",
        "__rfloordiv__",
        "__rmatmul__",
        "__rmod__",
        "__rmul__",
        "__ror__",
        "__rpow__",
        "__rsub__",
        "__rtruediv__",
        "__rxor__",
        "__setitem__",
        "__sub__",
        "__truediv__",
        "__xor__",
    ]

    for magic_method_name in _magic_method_names:
        setattr(
            XorbitsData,
            magic_method_name,
            wrap_magic_method(magic_method_name, pass_through=False),
        )
    setattr(XorbitsData, "__setattr__", wrap_setattr(pass_through=False))

    for magic_method_name in _magic_method_names:
        setattr(
            XorbitsDataRef,
            magic_method_name,
            wrap_magic_method(magic_method_name, pass_through=True),
        )
    setattr(XorbitsDataRef, "__setattr__", wrap_setattr(pass_through=True))


_install_magic_methods()
del _install_magic_methods
