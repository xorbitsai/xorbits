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

from ...adapter.mars import MarsEntity, mars_dataframe
from ...dataframe import DataFrame, get_dummies
from ..data import XorbitsDataRef, from_mars, to_mars
from ..execution import need_to_execute


def test_to_mars():
    xdf = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    assert isinstance(to_mars(xdf), MarsEntity)

    # tuple.
    ins = (xdf, "foo")
    assert isinstance(to_mars(ins)[0], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    # nested tuple.
    ins = ((xdf, "foo"), "bar")
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = ([xdf, "foo"], "bar")
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = ({"foo": xdf, "bar": "baz"}, "bar")
    assert isinstance(to_mars(ins)[0]["foo"], MarsEntity)
    assert not isinstance(to_mars(ins)[0]["bar"], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)

    # list.
    ins = [xdf, "foo"]
    assert isinstance(to_mars(ins)[0], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    # nested list.
    ins = [(xdf, "foo"), "bar"]
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = [[xdf, "foo"], "bar"]
    assert isinstance(to_mars(ins)[0][0], MarsEntity)
    assert not isinstance(to_mars(ins)[0][1], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)
    ins = [{"foo": xdf, "bar": "baz"}, "bar"]
    assert isinstance(to_mars(ins)[0]["foo"], MarsEntity)
    assert not isinstance(to_mars(ins)[0]["bar"], MarsEntity)
    assert not isinstance(to_mars(ins)[1], MarsEntity)

    # dict.
    ins = {"foo": xdf, "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)
    # nested dict.
    ins = {"foo": (xdf, "foo"), "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"][0], MarsEntity)
    assert not isinstance(to_mars(ins)["foo"][1], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)
    ins = {"foo": [xdf, "foo"], "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"][0], MarsEntity)
    assert not isinstance(to_mars(ins)["foo"][1], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)
    ins = {"foo": {"bar": xdf}, "bar": "baz"}
    assert isinstance(to_mars(ins)["foo"]["bar"], MarsEntity)
    assert not isinstance(to_mars(ins)["bar"], MarsEntity)


def test_from_mars():
    mdf = mars_dataframe.DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    assert isinstance(from_mars(mdf), XorbitsDataRef)

    # tuple.
    ins = (mdf, "foo")
    assert isinstance(from_mars(ins)[0], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[1], XorbitsDataRef)
    # nested tuple.
    ins = ((mdf, "foo"), "bar")
    assert isinstance(from_mars(ins)[0][0], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[0][1], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[1], XorbitsDataRef)
    ins = ([mdf, "foo"], "bar")
    assert isinstance(from_mars(ins)[0][0], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[0][1], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[1], XorbitsDataRef)
    ins = ({"foo": mdf, "bar": "baz"}, "bar")
    assert isinstance(from_mars(ins)[0]["foo"], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[0]["bar"], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[1], XorbitsDataRef)

    # list.
    ins = [mdf, "foo"]
    assert isinstance(from_mars(ins)[0], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[1], XorbitsDataRef)
    # nested list.
    ins = [(mdf, "foo"), "bar"]
    assert isinstance(from_mars(ins)[0][0], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[0][1], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[1], XorbitsDataRef)
    ins = [[mdf, "foo"], "bar"]
    assert isinstance(from_mars(ins)[0][0], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[0][1], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[1], XorbitsDataRef)
    ins = [{"foo": mdf, "bar": "baz"}, "bar"]
    assert isinstance(from_mars(ins)[0]["foo"], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[0]["bar"], XorbitsDataRef)
    assert not isinstance(from_mars(ins)[1], XorbitsDataRef)

    # dict.
    ins = {"foo": mdf, "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"], XorbitsDataRef)
    assert not isinstance(from_mars(ins)["bar"], XorbitsDataRef)
    # nested dict.
    ins = {"foo": (mdf, "foo"), "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"][0], XorbitsDataRef)
    assert not isinstance(from_mars(ins)["foo"][1], XorbitsDataRef)
    assert not isinstance(from_mars(ins)["bar"], XorbitsDataRef)
    ins = {"foo": [mdf, "foo"], "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"][0], XorbitsDataRef)
    assert not isinstance(from_mars(ins)["foo"][1], XorbitsDataRef)
    assert not isinstance(from_mars(ins)["bar"], XorbitsDataRef)
    ins = {"foo": {"bar": mdf}, "bar": "baz"}
    assert isinstance(from_mars(ins)["foo"]["bar"], XorbitsDataRef)
    assert not isinstance(from_mars(ins)["bar"], XorbitsDataRef)


def test_deferred_execution():
    # print.
    xdf = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    assert need_to_execute(xdf.proxied.proxied)
    print(xdf)
    assert not need_to_execute(xdf.proxied.proxied)

    # iterrows.
    xdf = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    assert need_to_execute(xdf.proxied.proxied)
    count = 0
    for (_, _) in xdf.iterrows():
        count += 1
    assert not need_to_execute(xdf.proxied.proxied)

    # transpose.
    xdf = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    xdf = xdf.transpose()
    assert need_to_execute(xdf.proxied.proxied)
    print(xdf)
    assert not need_to_execute(xdf.proxied.proxied)

    # get dummies.
    xdf = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    xdf = get_dummies(xdf)
    assert need_to_execute(xdf.proxied.proxied)
    print(xdf)
    assert not need_to_execute(xdf.proxied.proxied)

    # groupby apply
    xdf = DataFrame({"foo": (1, 2, 3), "bar": (4, 5, 6)})
    xdf = xdf.groupby("foo").apply(lambda df: df.sum())
    assert need_to_execute(xdf.proxied.proxied)
    print(xdf)
    assert not need_to_execute(xdf.proxied.proxied)
