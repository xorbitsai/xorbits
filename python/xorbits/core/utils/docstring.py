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

import inspect
from types import ModuleType
from typing import Any, Callable, List, Optional, Type


def add_docstring_disclaimer(
    docstring_src_module: Optional[ModuleType],
    docstring_src: Optional[Callable],
    doc: Optional[str],
) -> Optional[str]:
    if doc is None:
        return None

    if (
        docstring_src is not None
        and hasattr(docstring_src, "__module__")
        and docstring_src.__module__
    ):
        return (
            doc
            + f"\n\nThis docstring was copied from {docstring_src.__module__}.{docstring_src.__name__}"
        )
    elif docstring_src_module is not None:
        return (
            doc + f"\n\nThis docstring was copied from {docstring_src_module.__name__}"
        )
    else:
        return doc


def skip_doctest(doc: Optional[str]) -> Optional[str]:
    def skip_line(line):
        # NumPy docstring contains cursor and comment only example
        stripped = line.strip()
        if stripped == ">>>" or stripped.startswith(">>> #"):
            return line
        elif ">>>" in stripped and "+SKIP" not in stripped:
            if "# doctest:" in line:
                return line + ", +SKIP"
            else:
                return line + "  # doctest: +SKIP"
        else:
            return line

    if doc is None:
        return None
    return "\n".join([skip_line(line) for line in doc.split("\n")])


def add_arg_disclaimer(
    src: Optional[Any],
    dest: Optional[Any],
    doc: Optional[str],
) -> Optional[str]:
    import re

    def get_named_args(func: Callable) -> List[str]:
        s = inspect.signature(func)
        return [
            n
            for n, p in s.parameters.items()
            if p.kind in [p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY, p.KEYWORD_ONLY]
        ]

    def mark_unsupported_args(doc: str, args: List[str]) -> str:
        import re

        lines = doc.split("\n")
        for arg in args:
            subset = [
                (i, line)
                for i, line in enumerate(lines)
                if re.match(r"^\s*" + arg + " ?:", line)
            ]
            if len(subset) == 1:
                [(i, line)] = subset
                lines[i] = line + "  (Not supported yet)"
        return "\n".join(lines)

    def add_extra_args(doc: str, original_doc: Optional[Any], args: List[str]) -> str:
        def count_leading_spaces(s: str):
            return len(s) - len(s.lstrip(" "))

        def get_param_name(s: str) -> Optional[str]:
            # example:
            # '    data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame'
            m = re.match(r"\s*(\S+)\s*:.*", s)
            if m:
                return m.group(1)
            return None

        def is_param_description(s: str, num_leading_spaces: int) -> bool:
            if s.strip():
                return count_leading_spaces(s) == num_leading_spaces + 4
            else:
                return True

        if original_doc is None:
            return doc

        new_section = [""]

        lines = original_doc.splitlines()
        idx = 0
        parameter_pattern = re.compile(r"\s+Parameters")
        while idx < len(lines):
            if re.match(parameter_pattern, lines[idx]):
                num_leading_spaces = count_leading_spaces(lines[idx])
                new_section.append(" " * num_leading_spaces + "Extra Parameters")
                new_section.append(
                    " " * num_leading_spaces + "-" * len("Extra Parameters")
                )
                idx += 2
                while idx < len(lines) and get_param_name(lines[idx]) is not None:
                    param_name = get_param_name(lines[idx])
                    if param_name in args:
                        new_section.append(
                            " " * num_leading_spaces + lines[idx].strip()
                        )
                        idx += 1
                        while idx < len(lines) and is_param_description(
                            lines[idx], num_leading_spaces
                        ):
                            if lines[idx].strip():
                                new_section.append(
                                    " " * (num_leading_spaces + 4) + lines[idx].strip()
                                )
                            idx += 1
                    else:
                        idx += 1
                        while idx < len(lines) and is_param_description(
                            lines[idx], num_leading_spaces
                        ):
                            idx += 1
                new_section.append(" " * num_leading_spaces)
                break
            idx += 1

        return doc + "\n".join(new_section)

    if doc is None:
        return None
    if src is None or dest is None:
        return doc

    try:
        src_args = get_named_args(src)
        dest_args = get_named_args(dest)
        unsupported_args = [a for a in src_args if a not in dest_args]
        extra_args = [a for a in dest_args if a not in src_args]
        if unsupported_args:
            doc = mark_unsupported_args(doc, unsupported_args)
        if extra_args:
            doc = add_extra_args(doc, getattr(dest, "__doc__", None), extra_args)
    except ValueError:
        return doc

    return doc


def gen_docstring(
    docstring_src: Optional[Type],
    method_name: str,
) -> Optional[str]:
    if docstring_src is None:
        return None

    src_method = getattr(docstring_src, method_name, None)
    if src_method is None:
        return None

    doc = getattr(src_method, "__doc__", None)
    if isinstance(src_method, property):
        # some things like SeriesGroupBy.unique are generated.
        src_method = src_method.fget
        if not doc:
            doc = getattr(src_method, "__doc__", None)

    # pandas DataFrame/Series sometimes override methods without setting __doc__.
    if doc is None and docstring_src.__name__ in {"DataFrame", "Series"}:
        for cls in docstring_src.mro():
            src_method = getattr(cls, method_name, None)
            if src_method is not None and hasattr(src_method, "__doc__"):
                doc = src_method.__doc__
    return doc


def attach_docstring(
    docstring_src_module: ModuleType, docstring_src: Optional[Callable], func: Callable
) -> Callable:
    if docstring_src is None:
        func.__doc__ = ""
        return func

    doc = getattr(docstring_src, "__doc__", None)
    doc = skip_doctest(doc)
    doc = add_arg_disclaimer(docstring_src, func, doc)
    doc = add_docstring_disclaimer(docstring_src_module, docstring_src, doc)
    func.__doc__ = "" if doc is None else doc
    return func
