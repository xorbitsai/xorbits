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

import inspect
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy
import pandas

from ...core.data import DataType

_DATA_TYPE_TO_DOCSTRING_SRC: Dict[DataType, Tuple[ModuleType, Type]] = {
    DataType.dataframe: (pandas, pandas.DataFrame),
    DataType.series: (pandas, pandas.Series),
    DataType.index: (pandas, pandas.Index),
    DataType.categorical: (pandas, pandas.Categorical),
    DataType.dataframe_groupby: (
        pandas,
        pandas.core.groupby.generic.DataFrameGroupBy,
    ),
    DataType.series_groupby: (pandas, pandas.core.groupby.generic.SeriesGroupBy),
    DataType.tensor: (numpy, numpy.ndarray),
}


def get_base_indentation(doc: str) -> Optional[str]:
    # the layout of a pandas docstring is:
    # \n<base indentation><line>\n\n<base indentation><line>...
    # the layout of a numpy docstring is:
    # <line>\n\n<base indentation><line>...
    # thus, we need to find a line that starts with '\n\n' and the number of its leading
    # spaces would be the base indentation.
    idx = doc.find("\n\n")
    if idx + 2 < len(doc):
        doc = doc[idx + 2 :]
        return " " * (len(doc) - len(doc.lstrip(" ")))
    else:  # pragma: no cover
        return None


def add_docstring_disclaimer(
    docstring_src_module: Optional[ModuleType],
    docstring_src_cls: Optional[Callable],
    doc: Optional[str],
    fallback_warning: bool = False,
) -> Optional[str]:
    if doc is None:
        return None

    base_indentation = get_base_indentation(doc)
    if base_indentation is None:  # pragma: no cover
        return doc

    warning_msg = (
        f"\n\n{base_indentation}.. warning:: This method has not been implemented yet. Xorbits will try to "
        f"execute it with {docstring_src_module.__name__}."
        if fallback_warning and docstring_src_module
        else "\n"
    )

    if (
        docstring_src_cls is not None
        and hasattr(docstring_src_cls, "__module__")
        and docstring_src_cls.__module__
    ):
        return (
            doc + warning_msg + f"\n{base_indentation}This docstring was copied from "
            f"{docstring_src_cls.__module__}.{docstring_src_cls.__name__}."
        )
    elif docstring_src_module is not None:
        return (
            doc
            + warning_msg
            + f"\n{base_indentation}This docstring was copied from {docstring_src_module.__name__}."
        )
    else:
        return doc


def add_version_disclaimer(
    doc: Optional[str], docstring_src_module: Optional[ModuleType]
) -> Optional[str]:
    if doc is None or docstring_src_module is None:
        return doc

    lines = []
    for line in doc.splitlines():
        stripped = line.strip()
        if (
            stripped.startswith(".. deprecated::")
            or stripped.startswith(".. versionchanged::")
            or stripped.startswith(".. versionadded::")
        ):
            line = line + f"({docstring_src_module.__name__})"
        lines.append(line)

    return "\n".join(lines)


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
    return "\n".join([skip_line(line) for line in doc.splitlines()])


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

        def is_param_description(s: str, base_indentation: str) -> bool:
            if s.strip():
                return count_leading_spaces(s) == len(base_indentation) + 4
            else:
                return True

        if original_doc is None:
            return doc

        # the indentation of generated docstring must be consistent.
        base_indentation = get_base_indentation(doc)
        original_base_indentation = get_base_indentation(original_doc)
        if (
            base_indentation is None or original_base_indentation is None
        ):  # pragma: no cover
            return doc
        param_description_indentation = base_indentation + " " * 4

        new_section: List[str] = []
        lines = original_doc.splitlines()
        idx = 0
        parameter_pattern = re.compile(r"\s+Parameters")
        while idx < len(lines):
            if re.match(parameter_pattern, lines[idx]):
                idx += 2
                while idx < len(lines) and get_param_name(lines[idx]) is not None:
                    param_name = get_param_name(lines[idx])
                    if param_name in args:
                        if not new_section:
                            # init new section only if the original docstring has description of
                            # any extra parameter.
                            new_section.append("")
                            new_section.append(base_indentation + "Extra Parameters")
                            new_section.append(
                                base_indentation + "-" * len("Extra Parameters")
                            )
                        new_section.append(base_indentation + lines[idx].strip())
                        idx += 1
                        while idx < len(lines) and is_param_description(
                            lines[idx], original_base_indentation
                        ):
                            if lines[idx].strip():
                                new_section.append(
                                    param_description_indentation + lines[idx].strip()
                                )
                            idx += 1
                    else:
                        # skip the parameter description.
                        idx += 1
                        while idx < len(lines) and is_param_description(
                            lines[idx], original_base_indentation
                        ):
                            idx += 1
                new_section.append(base_indentation)
                break
            idx += 1

        return doc + "\n".join(new_section)

    if doc is None:
        return None
    if src is None or not callable(src) or dest is None or not callable(dest):
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


def gen_member_docstring(
    docstring_src_cls: Optional[Type],
    member_name: str,
) -> Optional[str]:
    if docstring_src_cls is None:
        return None

    src = getattr(docstring_src_cls, member_name, None)
    if src is None:
        return None

    doc = getattr(src, "__doc__", None)
    if isinstance(src, property):
        # some things like SeriesGroupBy.unique are generated.
        src = src.fget
        if not doc:
            doc = getattr(src, "__doc__", None)

    # pandas DataFrame/Series sometimes override methods without setting __doc__.
    if doc is None and docstring_src_cls.__name__ in {"DataFrame", "Series"}:
        for cls in docstring_src_cls.mro():
            src = getattr(cls, member_name, None)
            if src is not None and hasattr(src, "__doc__"):
                doc = src.__doc__
    return doc


def attach_module_callable_docstring(
    c: Callable,
    docstring_src_module: ModuleType,
    docstring_src: Optional[Callable],
    fallback_warning: bool = False,
) -> Callable:
    """
    Attach docstring to functions and constructors.
    """

    if docstring_src is None:
        c.__doc__ = ""
        return c

    doc = getattr(docstring_src, "__doc__", None)
    doc = skip_doctest(doc)
    doc = add_version_disclaimer(doc, docstring_src_module)
    doc = add_arg_disclaimer(docstring_src, c, doc)
    doc = add_docstring_disclaimer(docstring_src_module, None, doc, fallback_warning)
    c.__doc__ = "" if doc is None else doc
    return c


def attach_cls_member_docstring(
    member: Any,
    member_name: str,
    data_type: Optional[DataType] = None,
    docstring_src_module: Optional[ModuleType] = None,
    docstring_src_cls: Optional[Type] = None,
    fallback_warning: bool = False,
) -> Any:
    """
    Attach docstring to class members.
    """

    if (
        docstring_src_module is None
        and docstring_src_cls is None
        and data_type is not None
    ):
        docstring_src_module, docstring_src_cls = _DATA_TYPE_TO_DOCSTRING_SRC.get(
            data_type, (None, None)
        )
    doc = gen_member_docstring(docstring_src_cls, member_name)
    doc = skip_doctest(doc)
    doc = add_version_disclaimer(doc, docstring_src_module)
    docstring_src = getattr(docstring_src_cls, member_name, None)
    doc = add_arg_disclaimer(docstring_src, member, doc)
    doc = add_docstring_disclaimer(
        docstring_src_module, docstring_src_cls, doc, fallback_warning
    )
    member.__doc__ = "" if doc is None else doc
    return member
