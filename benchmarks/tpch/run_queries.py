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

import argparse
import functools
import inspect
import json
import time
from typing import Callable, List, Dict

import xorbits
import xorbits.pandas as xd


@functools.lru_cache
def load_lineitem(
    data_folder: str, dtype_backend: str = None, gpu: bool = False, **storage_options
) -> xd.DataFrame:
    data_path = data_folder + "/lineitem"
    df = xd.read_parquet(
        data_path,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        gpu=gpu,
    )
    df["L_SHIPDATE"] = xd.to_datetime(df.L_SHIPDATE, format="%Y-%m-%d")
    df["L_RECEIPTDATE"] = xd.to_datetime(df.L_RECEIPTDATE, format="%Y-%m-%d")
    df["L_COMMITDATE"] = xd.to_datetime(df.L_COMMITDATE, format="%Y-%m-%d")
    return df


@functools.lru_cache
def load_part(
    data_folder: str, dtype_backend: str = None, gpu: bool = False, **storage_options
) -> xd.DataFrame:
    data_path = data_folder + "/part"
    df = xd.read_parquet(
        data_path,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        gpu=gpu,
    )
    return df


@functools.lru_cache
def load_orders(
    data_folder: str, dtype_backend: str = None, gpu: bool = False, **storage_options
) -> xd.DataFrame:
    data_path = data_folder + "/orders"
    df = xd.read_parquet(
        data_path,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        gpu=gpu,
    )
    df["O_ORDERDATE"] = xd.to_datetime(df.O_ORDERDATE, format="%Y-%m-%d")
    return df


@functools.lru_cache
def load_customer(
    data_folder: str, dtype_backend: str = None, gpu: bool = False, **storage_options
) -> xd.DataFrame:
    data_path = data_folder + "/customer"
    df = xd.read_parquet(
        data_path,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        gpu=gpu,
    )
    return df


@functools.lru_cache
def load_nation(
    data_folder: str, dtype_backend: str = None, gpu: bool = False, **storage_options
) -> xd.DataFrame:
    data_path = data_folder + "/nation"
    df = xd.read_parquet(
        data_path,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        gpu=gpu,
    )
    return df


@functools.lru_cache
def load_region(
    data_folder: str, dtype_backend: str = None, gpu: bool = False, **storage_options
) -> xd.DataFrame:
    data_path = data_folder + "/region"
    df = xd.read_parquet(
        data_path,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        gpu=gpu,
    )
    return df


@functools.lru_cache
def load_supplier(
    data_folder: str, dtype_backend: str = None, gpu: bool = False, **storage_options
) -> xd.DataFrame:
    data_path = data_folder + "/supplier"
    df = xd.read_parquet(
        data_path,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        gpu=gpu,
    )
    return df


@functools.lru_cache
def load_partsupp(
    data_folder: str, dtype_backend: str = None, gpu: bool = False, **storage_options
) -> xd.DataFrame:
    data_path = data_folder + "/partsupp"
    df = xd.read_parquet(
        data_path,
        dtype_backend=dtype_backend,
        storage_options=storage_options,
        gpu=gpu,
    )
    return df


def timethis(q: Callable):
    @functools.wraps(q)
    def wrapped(*args, **kwargs):
        t = time.time()
        q(*args, **kwargs)
        print("%s Execution time (s): %f" % (q.__name__.upper(), time.time() - t))

    return wrapped


_query_to_datasets: Dict[int, List[str]] = dict()


def collect_datasets(func: Callable):
    _query_to_datasets[int(func.__name__[1:])] = list(
        inspect.signature(func).parameters
    )
    return func


@timethis
@collect_datasets
def q01(lineitem: xd.DataFrame):
    date = xd.Timestamp("1998-09-02")
    lineitem_filtered = lineitem.loc[
        :,
        [
            "L_QUANTITY",
            "L_EXTENDEDPRICE",
            "L_DISCOUNT",
            "L_TAX",
            "L_RETURNFLAG",
            "L_LINESTATUS",
            "L_SHIPDATE",
            "L_ORDERKEY",
        ],
    ]
    sel = lineitem_filtered.L_SHIPDATE <= date
    lineitem_filtered = lineitem_filtered[sel]
    lineitem_filtered["AVG_QTY"] = lineitem_filtered.L_QUANTITY
    lineitem_filtered["AVG_PRICE"] = lineitem_filtered.L_EXTENDEDPRICE
    lineitem_filtered["DISC_PRICE"] = lineitem_filtered.L_EXTENDEDPRICE * (
        1 - lineitem_filtered.L_DISCOUNT
    )
    lineitem_filtered["CHARGE"] = (
        lineitem_filtered.L_EXTENDEDPRICE
        * (1 - lineitem_filtered.L_DISCOUNT)
        * (1 + lineitem_filtered.L_TAX)
    )
    gb = lineitem_filtered.groupby(["L_RETURNFLAG", "L_LINESTATUS"], as_index=False)[
        [
            "L_QUANTITY",
            "L_EXTENDEDPRICE",
            "DISC_PRICE",
            "CHARGE",
            "AVG_QTY",
            "AVG_PRICE",
            "L_DISCOUNT",
            "L_ORDERKEY",
        ]
    ]
    total = gb.agg(
        {
            "L_QUANTITY": "sum",
            "L_EXTENDEDPRICE": "sum",
            "DISC_PRICE": "sum",
            "CHARGE": "sum",
            "AVG_QTY": "mean",
            "AVG_PRICE": "mean",
            "L_DISCOUNT": "mean",
            "L_ORDERKEY": "count",
        }
    )
    # skip sort, Mars groupby enables sort
    # total = total.sort_values(["L_RETURNFLAG", "L_LINESTATUS"])
    print(total)


@timethis
@collect_datasets
def q02(part, partsupp, supplier, nation, region):
    nation_filtered = nation.loc[:, ["N_NATIONKEY", "N_NAME", "N_REGIONKEY"]]
    region_filtered = region[(region["R_NAME"] == "EUROPE")]
    region_filtered = region_filtered.loc[:, ["R_REGIONKEY"]]
    r_n_merged = nation_filtered.merge(
        region_filtered, left_on="N_REGIONKEY", right_on="R_REGIONKEY", how="inner"
    )
    r_n_merged = r_n_merged.loc[:, ["N_NATIONKEY", "N_NAME"]]
    supplier_filtered = supplier.loc[
        :,
        [
            "S_SUPPKEY",
            "S_NAME",
            "S_ADDRESS",
            "S_NATIONKEY",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
        ],
    ]
    s_r_n_merged = r_n_merged.merge(
        supplier_filtered, left_on="N_NATIONKEY", right_on="S_NATIONKEY", how="inner"
    )
    s_r_n_merged = s_r_n_merged.loc[
        :,
        [
            "N_NAME",
            "S_SUPPKEY",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
        ],
    ]
    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY", "PS_SUPPLYCOST"]]
    ps_s_r_n_merged = s_r_n_merged.merge(
        partsupp_filtered, left_on="S_SUPPKEY", right_on="PS_SUPPKEY", how="inner"
    )
    ps_s_r_n_merged = ps_s_r_n_merged.loc[
        :,
        [
            "N_NAME",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
            "PS_PARTKEY",
            "PS_SUPPLYCOST",
        ],
    ]
    part_filtered = part.loc[:, ["P_PARTKEY", "P_MFGR", "P_SIZE", "P_TYPE"]]
    part_filtered = part_filtered[
        (part_filtered["P_SIZE"] == 15)
        & (part_filtered["P_TYPE"].str.endswith("BRASS"))
    ]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY", "P_MFGR"]]
    merged_df = part_filtered.merge(
        ps_s_r_n_merged, left_on="P_PARTKEY", right_on="PS_PARTKEY", how="inner"
    )
    merged_df = merged_df.loc[
        :,
        [
            "N_NAME",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
            "PS_SUPPLYCOST",
            "P_PARTKEY",
            "P_MFGR",
        ],
    ]
    min_values = merged_df.groupby("P_PARTKEY", as_index=False, sort=False)[
        "PS_SUPPLYCOST"
    ].min()
    min_values.columns = ["P_PARTKEY_CPY", "MIN_SUPPLYCOST"]
    merged_df = merged_df.merge(
        min_values,
        left_on=["P_PARTKEY", "PS_SUPPLYCOST"],
        right_on=["P_PARTKEY_CPY", "MIN_SUPPLYCOST"],
        how="inner",
    )
    total = merged_df.loc[
        :,
        [
            "S_ACCTBAL",
            "S_NAME",
            "N_NAME",
            "P_PARTKEY",
            "P_MFGR",
            "S_ADDRESS",
            "S_PHONE",
            "S_COMMENT",
        ],
    ]
    total = total.sort_values(
        by=["S_ACCTBAL", "N_NAME", "S_NAME", "P_PARTKEY"],
        ascending=[False, True, True, True],
    )
    print(total.head(100))


@timethis
@collect_datasets
def q03(lineitem, orders, customer):
    date = xd.Timestamp("1995-03-15")
    lineitem_filtered = lineitem.loc[
        :, ["L_ORDERKEY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE"]
    ]
    orders_filtered = orders.loc[
        :, ["O_ORDERKEY", "O_CUSTKEY", "O_ORDERDATE", "O_SHIPPRIORITY"]
    ]
    customer_filtered = customer.loc[:, ["C_MKTSEGMENT", "C_CUSTKEY"]]
    lsel = lineitem_filtered.L_SHIPDATE > date
    osel = orders_filtered.O_ORDERDATE < date
    csel = customer_filtered.C_MKTSEGMENT == "BUILDING"
    flineitem = lineitem_filtered[lsel]
    forders = orders_filtered[osel]
    fcustomer = customer_filtered[csel]
    jn1 = fcustomer.merge(forders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn2 = jn1.merge(flineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn2["TMP"] = jn2.L_EXTENDEDPRICE * (1 - jn2.L_DISCOUNT)
    total = (
        jn2.groupby(
            ["L_ORDERKEY", "O_ORDERDATE", "O_SHIPPRIORITY"], as_index=False, sort=False
        )["TMP"]
        .sum()
        .sort_values(["TMP"], ascending=False)
    )
    res = total.loc[:, ["L_ORDERKEY", "TMP", "O_ORDERDATE", "O_SHIPPRIORITY"]]
    print(res.head(10))


@timethis
@collect_datasets
def q04(lineitem, orders):
    date1 = xd.Timestamp("1993-10-01")
    date2 = xd.Timestamp("1993-07-01")
    lsel = lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE
    osel = (orders.O_ORDERDATE < date1) & (orders.O_ORDERDATE >= date2)
    flineitem = lineitem[lsel]
    forders = orders[osel]
    jn = forders[forders["O_ORDERKEY"].isin(flineitem["L_ORDERKEY"])]
    total = jn.groupby("O_ORDERPRIORITY", as_index=False)["O_ORDERKEY"].count()
    print(total)


@timethis
@collect_datasets
def q05(lineitem, orders, customer, nation, region, supplier):
    date1 = xd.Timestamp("1994-01-01")
    date2 = xd.Timestamp("1995-01-01")
    rsel = region.R_NAME == "ASIA"
    osel = (orders.O_ORDERDATE >= date1) & (orders.O_ORDERDATE < date2)
    forders = orders[osel]
    fregion = region[rsel]
    jn1 = fregion.merge(nation, left_on="R_REGIONKEY", right_on="N_REGIONKEY")
    jn2 = jn1.merge(customer, left_on="N_NATIONKEY", right_on="C_NATIONKEY")
    jn3 = jn2.merge(forders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn4 = jn3.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn5 = supplier.merge(
        jn4, left_on=["S_SUPPKEY", "S_NATIONKEY"], right_on=["L_SUPPKEY", "N_NATIONKEY"]
    )
    jn5["TMP"] = jn5.L_EXTENDEDPRICE * (1.0 - jn5.L_DISCOUNT)
    gb = jn5.groupby("N_NAME", as_index=False, sort=False)["TMP"].sum()
    total = gb.sort_values("TMP", ascending=False)
    print(total)


@timethis
@collect_datasets
def q06(lineitem):
    date1 = xd.Timestamp("1994-01-01")
    date2 = xd.Timestamp("1995-01-01")
    lineitem_filtered = lineitem.loc[
        :, ["L_QUANTITY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE"]
    ]
    sel = (
        (lineitem_filtered.L_SHIPDATE >= date1)
        & (lineitem_filtered.L_SHIPDATE < date2)
        & (lineitem_filtered.L_DISCOUNT >= 0.05)
        & (lineitem_filtered.L_DISCOUNT <= 0.07)
        & (lineitem_filtered.L_QUANTITY < 24)
    )
    flineitem = lineitem_filtered[sel]
    total = (flineitem.L_EXTENDEDPRICE * flineitem.L_DISCOUNT).sum()
    print(total)


@timethis
@collect_datasets
def q07(lineitem, supplier, orders, customer, nation):
    """This version is faster than q07_old. Keeping the old one for reference"""
    lineitem_filtered = lineitem[
        (lineitem["L_SHIPDATE"] >= xd.Timestamp("1995-01-01"))
        & (lineitem["L_SHIPDATE"] < xd.Timestamp("1997-01-01"))
    ]
    lineitem_filtered["L_YEAR"] = lineitem_filtered["L_SHIPDATE"].dt.year
    lineitem_filtered["VOLUME"] = lineitem_filtered["L_EXTENDEDPRICE"] * (
        1.0 - lineitem_filtered["L_DISCOUNT"]
    )
    lineitem_filtered = lineitem_filtered.loc[
        :, ["L_ORDERKEY", "L_SUPPKEY", "L_YEAR", "VOLUME"]
    ]
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    orders_filtered = orders.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
    customer_filtered = customer.loc[:, ["C_CUSTKEY", "C_NATIONKEY"]]
    n1 = nation[(nation["N_NAME"] == "FRANCE")].loc[:, ["N_NATIONKEY", "N_NAME"]]
    n2 = nation[(nation["N_NAME"] == "GERMANY")].loc[:, ["N_NATIONKEY", "N_NAME"]]

    # ----- do nation 1 -----
    N1_C = customer_filtered.merge(
        n1, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N1_C = N1_C.drop(columns=["C_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "CUST_NATION"}
    )
    N1_C_O = N1_C.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="inner"
    )
    N1_C_O = N1_C_O.drop(columns=["C_CUSTKEY", "O_CUSTKEY"])

    N2_S = supplier_filtered.merge(
        n2, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N2_S = N2_S.drop(columns=["S_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "SUPP_NATION"}
    )
    N2_S_L = N2_S.merge(
        lineitem_filtered, left_on="S_SUPPKEY", right_on="L_SUPPKEY", how="inner"
    )
    N2_S_L = N2_S_L.drop(columns=["S_SUPPKEY", "L_SUPPKEY"])

    total1 = N1_C_O.merge(
        N2_S_L, left_on="O_ORDERKEY", right_on="L_ORDERKEY", how="inner"
    )
    total1 = total1.drop(columns=["O_ORDERKEY", "L_ORDERKEY"])

    # ----- do nation 2 -----
    N2_C = customer_filtered.merge(
        n2, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N2_C = N2_C.drop(columns=["C_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "CUST_NATION"}
    )
    N2_C_O = N2_C.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="inner"
    )
    N2_C_O = N2_C_O.drop(columns=["C_CUSTKEY", "O_CUSTKEY"])

    N1_S = supplier_filtered.merge(
        n1, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N1_S = N1_S.drop(columns=["S_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "SUPP_NATION"}
    )
    N1_S_L = N1_S.merge(
        lineitem_filtered, left_on="S_SUPPKEY", right_on="L_SUPPKEY", how="inner"
    )
    N1_S_L = N1_S_L.drop(columns=["S_SUPPKEY", "L_SUPPKEY"])

    total2 = N2_C_O.merge(
        N1_S_L, left_on="O_ORDERKEY", right_on="L_ORDERKEY", how="inner"
    )
    total2 = total2.drop(columns=["O_ORDERKEY", "L_ORDERKEY"])

    # concat results
    total = xd.concat([total1, total2])

    total = total.groupby(["SUPP_NATION", "CUST_NATION", "L_YEAR"], as_index=False).agg(
        REVENUE=xd.NamedAgg(column="VOLUME", aggfunc="sum")
    )
    # skip sort when Mars groupby does sort already
    # total = total.sort_values(
    #     by=["SUPP_NATION", "CUST_NATION", "L_YEAR"], ascending=[True, True, True]
    # )
    print(total)


@timethis
@collect_datasets
def q08(part, lineitem, supplier, orders, customer, nation, region):
    part_filtered = part[(part["P_TYPE"] == "ECONOMY ANODIZED STEEL")]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY"]]
    lineitem_filtered = lineitem.loc[:, ["L_PARTKEY", "L_SUPPKEY", "L_ORDERKEY"]]
    lineitem_filtered["VOLUME"] = lineitem["L_EXTENDEDPRICE"] * (
        1.0 - lineitem["L_DISCOUNT"]
    )
    total = part_filtered.merge(
        lineitem_filtered, left_on="P_PARTKEY", right_on="L_PARTKEY", how="inner"
    )
    total = total.loc[:, ["L_SUPPKEY", "L_ORDERKEY", "VOLUME"]]
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    total = total.merge(
        supplier_filtered, left_on="L_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    total = total.loc[:, ["L_ORDERKEY", "VOLUME", "S_NATIONKEY"]]
    orders_filtered = orders[
        (orders["O_ORDERDATE"] >= xd.Timestamp("1995-01-01"))
        & (orders["O_ORDERDATE"] < xd.Timestamp("1997-01-01"))
    ]
    orders_filtered["O_YEAR"] = orders_filtered["O_ORDERDATE"].dt.year
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY", "O_CUSTKEY", "O_YEAR"]]
    total = total.merge(
        orders_filtered, left_on="L_ORDERKEY", right_on="O_ORDERKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_CUSTKEY", "O_YEAR"]]
    customer_filtered = customer.loc[:, ["C_CUSTKEY", "C_NATIONKEY"]]
    total = total.merge(
        customer_filtered, left_on="O_CUSTKEY", right_on="C_CUSTKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_YEAR", "C_NATIONKEY"]]
    n1_filtered = nation.loc[:, ["N_NATIONKEY", "N_REGIONKEY"]]
    n2_filtered = nation.loc[:, ["N_NATIONKEY", "N_NAME"]].rename(
        columns={"N_NAME": "NATION"}
    )
    total = total.merge(
        n1_filtered, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_YEAR", "N_REGIONKEY"]]
    total = total.merge(
        n2_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "O_YEAR", "N_REGIONKEY", "NATION"]]
    region_filtered = region[(region["R_NAME"] == "AMERICA")]
    region_filtered = region_filtered.loc[:, ["R_REGIONKEY"]]
    total = total.merge(
        region_filtered, left_on="N_REGIONKEY", right_on="R_REGIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "O_YEAR", "NATION"]]

    def udf(df):
        demoniator = df["VOLUME"].sum()
        df = df[df["NATION"] == "BRAZIL"]
        numerator = df["VOLUME"].sum()
        return numerator / demoniator

    total = total.groupby("O_YEAR", as_index=False).apply(udf)
    total.columns = ["O_YEAR", "MKT_SHARE"]
    total = total.sort_values(by=["O_YEAR"], ascending=[True])
    print(total)


@timethis
@collect_datasets
def q09(lineitem, orders, part, nation, partsupp, supplier):
    psel = part.P_NAME.str.contains("green")
    fpart = part[psel]
    jn1 = lineitem.merge(fpart, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn2 = jn1.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
    jn3 = jn2.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
    jn4 = partsupp.merge(
        jn3, left_on=["PS_PARTKEY", "PS_SUPPKEY"], right_on=["L_PARTKEY", "L_SUPPKEY"]
    )
    jn5 = jn4.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn5["TMP"] = jn5.L_EXTENDEDPRICE * (1 - jn5.L_DISCOUNT) - (
        (1 * jn5.PS_SUPPLYCOST) * jn5.L_QUANTITY
    )
    jn5["O_YEAR"] = jn5.O_ORDERDATE.dt.year
    gb = jn5.groupby(["N_NAME", "O_YEAR"], as_index=False, sort=False)["TMP"].sum()
    total = gb.sort_values(["N_NAME", "O_YEAR"], ascending=[True, False])
    print(total)


@timethis
@collect_datasets
def q10(lineitem, orders, customer, nation):
    date1 = xd.Timestamp("1993-10-01")
    date2 = xd.Timestamp("1994-01-01")
    osel = (orders.O_ORDERDATE >= date1) & (orders.O_ORDERDATE < date2)
    lsel = lineitem.L_RETURNFLAG == "R"
    forders = orders[osel]
    flineitem = lineitem[lsel]
    jn1 = flineitem.merge(forders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    jn3 = jn2.merge(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn3["TMP"] = jn3.L_EXTENDEDPRICE * (1.0 - jn3.L_DISCOUNT)
    gb = jn3.groupby(
        [
            "C_CUSTKEY",
            "C_NAME",
            "C_ACCTBAL",
            "C_PHONE",
            "N_NAME",
            "C_ADDRESS",
            "C_COMMENT",
        ],
        as_index=False,
        sort=False,
    )["TMP"].sum()
    total = gb.sort_values("TMP", ascending=False)
    print(total.head(20))


@timethis
@collect_datasets
def q11(partsupp, supplier, nation):
    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY"]]
    partsupp_filtered["TOTAL_COST"] = (
        partsupp["PS_SUPPLYCOST"] * partsupp["PS_AVAILQTY"]
    )
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    ps_supp_merge = partsupp_filtered.merge(
        supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    ps_supp_merge = ps_supp_merge.loc[:, ["PS_PARTKEY", "S_NATIONKEY", "TOTAL_COST"]]
    nation_filtered = nation[(nation["N_NAME"] == "GERMANY")]
    nation_filtered = nation_filtered.loc[:, ["N_NATIONKEY"]]
    ps_supp_n_merge = ps_supp_merge.merge(
        nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    ps_supp_n_merge = ps_supp_n_merge.loc[:, ["PS_PARTKEY", "TOTAL_COST"]]
    sum_val = ps_supp_n_merge["TOTAL_COST"].sum() * 0.0001
    total = ps_supp_n_merge.groupby(["PS_PARTKEY"], as_index=False, sort=False).agg(
        VALUE=xd.NamedAgg(column="TOTAL_COST", aggfunc="sum")
    )
    total = total[total["VALUE"] > sum_val]
    total = total.sort_values("VALUE", ascending=False)
    print(total)


@timethis
@collect_datasets
def q12(lineitem, orders):
    date1 = xd.Timestamp("1994-01-01")
    date2 = xd.Timestamp("1995-01-01")
    sel = (
        (lineitem.L_RECEIPTDATE < date2)
        & (lineitem.L_COMMITDATE < date2)
        & (lineitem.L_SHIPDATE < date2)
        & (lineitem.L_SHIPDATE < lineitem.L_COMMITDATE)
        & (lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE)
        & (lineitem.L_RECEIPTDATE >= date1)
        & ((lineitem.L_SHIPMODE == "MAIL") | (lineitem.L_SHIPMODE == "SHIP"))
    )
    flineitem = lineitem[sel]
    jn = flineitem.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")

    def g1(x):
        return ((x == "1-URGENT") | (x == "2-HIGH")).sum()

    def g2(x):
        return ((x != "1-URGENT") & (x != "2-HIGH")).sum()

    total = jn.groupby("L_SHIPMODE", as_index=False)["O_ORDERPRIORITY"].agg((g1, g2))
    total = total.reset_index()  # reset index to keep consistency with pandas
    # skip sort when groupby does sort already
    # total = total.sort_values("L_SHIPMODE")
    print(total)


@timethis
@collect_datasets
def q13(customer, orders):
    customer_filtered = customer.loc[:, ["C_CUSTKEY"]]
    orders_filtered = orders[
        ~orders["O_COMMENT"].str.contains(r"special[\S|\s]*requests")
    ]
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
    c_o_merged = customer_filtered.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left"
    )
    c_o_merged = c_o_merged.loc[:, ["C_CUSTKEY", "O_ORDERKEY"]]
    count_df = c_o_merged.groupby(["C_CUSTKEY"], as_index=False, sort=False).agg(
        C_COUNT=xd.NamedAgg(column="O_ORDERKEY", aggfunc="count")
    )
    total = count_df.groupby(["C_COUNT"], as_index=False, sort=False).size()
    total.columns = ["C_COUNT", "CUSTDIST"]
    total = total.sort_values(by=["CUSTDIST", "C_COUNT"], ascending=[False, False])
    print(total)


@timethis
@collect_datasets
def q14(lineitem, part):
    startDate = xd.Timestamp("1995-09-01")
    endDate = xd.Timestamp("1995-10-01")
    p_type_like = "PROMO"
    part_filtered = part.loc[:, ["P_PARTKEY", "P_TYPE"]]
    lineitem_filtered = lineitem.loc[
        :, ["L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE", "L_PARTKEY"]
    ]
    sel = (lineitem_filtered.L_SHIPDATE >= startDate) & (
        lineitem_filtered.L_SHIPDATE < endDate
    )
    flineitem = lineitem_filtered[sel]
    jn = flineitem.merge(part_filtered, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn["TMP"] = jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)
    total = jn[jn.P_TYPE.str.startswith(p_type_like)].TMP.sum() * 100 / jn.TMP.sum()
    print(total)


@timethis
@collect_datasets
def q15(lineitem, supplier):
    lineitem_filtered = lineitem[
        (lineitem["L_SHIPDATE"] >= xd.Timestamp("1996-01-01"))
        & (
            lineitem["L_SHIPDATE"]
            < (xd.Timestamp("1996-01-01") + xd.DateOffset(months=3))
        )
    ]
    lineitem_filtered["REVENUE_PARTS"] = lineitem_filtered["L_EXTENDEDPRICE"] * (
        1.0 - lineitem_filtered["L_DISCOUNT"]
    )
    lineitem_filtered = lineitem_filtered.loc[:, ["L_SUPPKEY", "REVENUE_PARTS"]]
    revenue_table = (
        lineitem_filtered.groupby("L_SUPPKEY", as_index=False, sort=False)
        .agg(TOTAL_REVENUE=xd.NamedAgg(column="REVENUE_PARTS", aggfunc="sum"))
        .rename(columns={"L_SUPPKEY": "SUPPLIER_NO"})
    )
    max_revenue = revenue_table["TOTAL_REVENUE"].max()
    revenue_table = revenue_table[revenue_table["TOTAL_REVENUE"] == max_revenue]
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE"]]
    total = supplier_filtered.merge(
        revenue_table, left_on="S_SUPPKEY", right_on="SUPPLIER_NO", how="inner"
    )
    total = total.loc[
        :, ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE", "TOTAL_REVENUE"]
    ]
    print(total)


@timethis
@collect_datasets
def q16(part, partsupp, supplier):
    part_filtered = part[
        (part["P_BRAND"] != "Brand#45")
        & (~part["P_TYPE"].str.contains("^MEDIUM POLISHED"))
        & part["P_SIZE"].isin([49, 14, 23, 45, 19, 3, 36, 9])
    ]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY", "P_BRAND", "P_TYPE", "P_SIZE"]]
    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY"]]
    total = part_filtered.merge(
        partsupp_filtered, left_on="P_PARTKEY", right_on="PS_PARTKEY", how="inner"
    )
    total = total.loc[:, ["P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY"]]
    supplier_filtered = supplier[
        supplier["S_COMMENT"].str.contains(r"Customer(\S|\s)*Complaints")
    ]
    supplier_filtered = supplier_filtered.loc[:, ["S_SUPPKEY"]].drop_duplicates()
    # left merge to select only PS_SUPPKEY values not in supplier_filtered
    total = total.merge(
        supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="left"
    )
    total = total[total["S_SUPPKEY"].isna()]
    total = total.loc[:, ["P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY"]]
    total = total.groupby(["P_BRAND", "P_TYPE", "P_SIZE"], as_index=False, sort=False)[
        "PS_SUPPKEY"
    ].nunique()
    total.columns = ["P_BRAND", "P_TYPE", "P_SIZE", "SUPPLIER_CNT"]
    total = total.sort_values(
        by=["SUPPLIER_CNT", "P_BRAND", "P_TYPE", "P_SIZE"],
        ascending=[False, True, True, True],
    )
    print(total)


@timethis
@collect_datasets
def q17(lineitem, part):
    left = lineitem.loc[:, ["L_PARTKEY", "L_QUANTITY", "L_EXTENDEDPRICE"]]
    right = part[((part["P_BRAND"] == "Brand#23") & (part["P_CONTAINER"] == "MED BOX"))]
    right = right.loc[:, ["P_PARTKEY"]]
    line_part_merge = left.merge(
        right, left_on="L_PARTKEY", right_on="P_PARTKEY", how="inner"
    )
    line_part_merge = line_part_merge.loc[
        :, ["L_QUANTITY", "L_EXTENDEDPRICE", "P_PARTKEY"]
    ]
    lineitem_filtered = lineitem.loc[:, ["L_PARTKEY", "L_QUANTITY"]]
    lineitem_avg = lineitem_filtered.groupby(
        ["L_PARTKEY"], as_index=False, sort=False
    ).agg(avg=xd.NamedAgg(column="L_QUANTITY", aggfunc="mean"))
    lineitem_avg["avg"] = 0.2 * lineitem_avg["avg"]
    lineitem_avg = lineitem_avg.loc[:, ["L_PARTKEY", "avg"]]
    total = line_part_merge.merge(
        lineitem_avg, left_on="P_PARTKEY", right_on="L_PARTKEY", how="inner"
    )
    total = total[total["L_QUANTITY"] < total["avg"]]
    total = xd.DataFrame({"avg_yearly": [total["L_EXTENDEDPRICE"].sum() / 7.0]})
    print(total)


@timethis
@collect_datasets
def q18(lineitem, orders, customer):
    gb1 = lineitem.groupby("L_ORDERKEY", as_index=False, sort=False)["L_QUANTITY"].sum()
    fgb1 = gb1[gb1.L_QUANTITY > 300]
    jn1 = fgb1.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    gb2 = jn2.groupby(
        ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
        as_index=False,
        sort=False,
    )["L_QUANTITY"].sum()
    total = gb2.sort_values(["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True])
    print(total)


@timethis
@collect_datasets
def q19(lineitem, part):
    Brand12 = "Brand#12"
    Brand23 = "Brand#23"
    Brand34 = "Brand#34"
    SMBOX = "SM BOX"
    SMCASE = "SM CASE"
    SMPACK = "SM PACK"
    SMPKG = "SM PKG"
    MEDBAG = "MED BAG"
    MEDBOX = "MED BOX"
    MEDPACK = "MED PACK"
    MEDPKG = "MED PKG"
    LGBOX = "LG BOX"
    LGCASE = "LG CASE"
    LGPACK = "LG PACK"
    LGPKG = "LG PKG"
    DELIVERINPERSON = "DELIVER IN PERSON"
    AIR = "AIR"
    AIRREG = "AIRREG"
    flineitem = lineitem[
        ((lineitem.L_SHIPMODE == AIR) | (lineitem.L_SHIPMODE == AIRREG))
        & (lineitem.L_SHIPINSTRUCT == DELIVERINPERSON)
    ]
    jn = flineitem.merge(part, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jnsel = (
        (
            (jn.P_BRAND == Brand12)
            & (
                (jn.P_CONTAINER == SMCASE)
                | (jn.P_CONTAINER == SMBOX)
                | (jn.P_CONTAINER == SMPACK)
                | (jn.P_CONTAINER == SMPKG)
            )
            & (jn.L_QUANTITY >= 1)
            & (jn.L_QUANTITY <= 11)
            & (jn.P_SIZE >= 1)
            & (jn.P_SIZE <= 5)
        )
        | (
            (jn.P_BRAND == Brand23)
            & (
                (jn.P_CONTAINER == MEDBAG)
                | (jn.P_CONTAINER == MEDBOX)
                | (jn.P_CONTAINER == MEDPKG)
                | (jn.P_CONTAINER == MEDPACK)
            )
            & (jn.L_QUANTITY >= 10)
            & (jn.L_QUANTITY <= 20)
            & (jn.P_SIZE >= 1)
            & (jn.P_SIZE <= 10)
        )
        | (
            (jn.P_BRAND == Brand34)
            & (
                (jn.P_CONTAINER == LGCASE)
                | (jn.P_CONTAINER == LGBOX)
                | (jn.P_CONTAINER == LGPACK)
                | (jn.P_CONTAINER == LGPKG)
            )
            & (jn.L_QUANTITY >= 20)
            & (jn.L_QUANTITY <= 30)
            & (jn.P_SIZE >= 1)
            & (jn.P_SIZE <= 15)
        )
    )
    jn = jn[jnsel]
    total = (jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)).sum()
    print(total)


@timethis
@collect_datasets
def q20(lineitem, part, nation, partsupp, supplier):
    date1 = xd.Timestamp("1994-01-01")
    date2 = xd.Timestamp("1995-01-01")
    psel = part.P_NAME.str.startswith("forest")
    nsel = nation.N_NAME == "CANADA"
    lsel = (lineitem.L_SHIPDATE >= date1) & (lineitem.L_SHIPDATE < date2)
    fpart = part[psel]
    fnation = nation[nsel]
    flineitem = lineitem[lsel]
    jn1 = fpart.merge(partsupp, left_on="P_PARTKEY", right_on="PS_PARTKEY")
    jn2 = jn1.merge(
        flineitem,
        left_on=["PS_PARTKEY", "PS_SUPPKEY"],
        right_on=["L_PARTKEY", "L_SUPPKEY"],
    )
    gb = jn2.groupby(
        ["PS_PARTKEY", "PS_SUPPKEY", "PS_AVAILQTY"], as_index=False, sort=False
    )["L_QUANTITY"].sum()
    gbsel = gb.PS_AVAILQTY > (0.5 * gb.L_QUANTITY)
    fgb = gb[gbsel]
    jn3 = fgb.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
    jn4 = fnation.merge(jn3, left_on="N_NATIONKEY", right_on="S_NATIONKEY")
    jn4 = jn4.loc[:, ["S_NAME", "S_ADDRESS"]]
    total = jn4.sort_values("S_NAME").drop_duplicates()
    print(total)


@timethis
@collect_datasets
def q21(lineitem, orders, supplier, nation):
    lineitem_filtered = lineitem.loc[
        :, ["L_ORDERKEY", "L_SUPPKEY", "L_RECEIPTDATE", "L_COMMITDATE"]
    ]

    # Keep all rows that have another row in linetiem with the same orderkey and different suppkey
    lineitem_orderkeys = (
        lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY"]]
        .groupby("L_ORDERKEY", as_index=False, sort=False)["L_SUPPKEY"]
        .nunique()
    )
    lineitem_orderkeys.columns = ["L_ORDERKEY", "nunique_col"]
    lineitem_orderkeys = lineitem_orderkeys[lineitem_orderkeys["nunique_col"] > 1]
    lineitem_orderkeys = lineitem_orderkeys.loc[:, ["L_ORDERKEY"]]

    # Keep all rows that have l_receiptdate > l_commitdate
    lineitem_filtered = lineitem_filtered[
        lineitem_filtered["L_RECEIPTDATE"] > lineitem_filtered["L_COMMITDATE"]
    ]
    lineitem_filtered = lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY"]]

    # Merge Filter + Exists
    lineitem_filtered = lineitem_filtered.merge(
        lineitem_orderkeys, on="L_ORDERKEY", how="inner"
    )

    # Not Exists: Check the exists condition isn't still satisfied on the output.
    lineitem_orderkeys = lineitem_filtered.groupby(
        "L_ORDERKEY", as_index=False, sort=False
    )["L_SUPPKEY"].nunique()
    lineitem_orderkeys.columns = ["L_ORDERKEY", "nunique_col"]
    lineitem_orderkeys = lineitem_orderkeys[lineitem_orderkeys["nunique_col"] == 1]
    lineitem_orderkeys = lineitem_orderkeys.loc[:, ["L_ORDERKEY"]]

    # Merge Filter + Not Exists
    lineitem_filtered = lineitem_filtered.merge(
        lineitem_orderkeys, on="L_ORDERKEY", how="inner"
    )

    orders_filtered = orders.loc[:, ["O_ORDERSTATUS", "O_ORDERKEY"]]
    orders_filtered = orders_filtered[orders_filtered["O_ORDERSTATUS"] == "F"]
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY"]]
    total = lineitem_filtered.merge(
        orders_filtered, left_on="L_ORDERKEY", right_on="O_ORDERKEY", how="inner"
    )
    total = total.loc[:, ["L_SUPPKEY"]]

    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY", "S_NAME"]]
    total = total.merge(
        supplier_filtered, left_on="L_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    total = total.loc[:, ["S_NATIONKEY", "S_NAME"]]
    nation_filtered = nation.loc[:, ["N_NAME", "N_NATIONKEY"]]
    nation_filtered = nation_filtered[nation_filtered["N_NAME"] == "SAUDI ARABIA"]
    total = total.merge(
        nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["S_NAME"]]
    total = total.groupby("S_NAME", as_index=False, sort=False).size()
    total.columns = ["S_NAME", "NUMWAIT"]
    total = total.sort_values(by=["NUMWAIT", "S_NAME"], ascending=[False, True])
    print(total.head(100))


@timethis
@collect_datasets
def q22(customer, orders):
    customer_filtered = customer.loc[:, ["C_ACCTBAL", "C_CUSTKEY"]]
    customer_filtered["CNTRYCODE"] = customer["C_PHONE"].str.slice(0, 2)
    customer_filtered = customer_filtered[
        (customer["C_ACCTBAL"] > 0.00)
        & customer_filtered["CNTRYCODE"].isin(
            ["13", "31", "23", "29", "30", "18", "17"]
        )
    ]
    avg_value = customer_filtered["C_ACCTBAL"].mean()
    customer_filtered = customer_filtered[customer_filtered["C_ACCTBAL"] > avg_value]
    # Select only the keys that don't match by performing a left join and only selecting columns with an na value
    orders_filtered = orders.loc[:, ["O_CUSTKEY"]].drop_duplicates()
    customer_keys = customer_filtered.loc[:, ["C_CUSTKEY"]].drop_duplicates()
    customer_selected = customer_keys.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left"
    )
    customer_selected = customer_selected[customer_selected["O_CUSTKEY"].isna()]
    customer_selected = customer_selected.loc[:, ["C_CUSTKEY"]]
    customer_selected = customer_selected.merge(
        customer_filtered, on="C_CUSTKEY", how="inner"
    )
    customer_selected = customer_selected.loc[:, ["CNTRYCODE", "C_ACCTBAL"]]
    agg1 = customer_selected.groupby(["CNTRYCODE"], as_index=False, sort=False).size()
    agg1.columns = ["CNTRYCODE", "NUMCUST"]
    agg2 = customer_selected.groupby(["CNTRYCODE"], as_index=False, sort=False).agg(
        TOTACCTBAL=xd.NamedAgg(column="C_ACCTBAL", aggfunc="sum")
    )
    total = agg1.merge(agg2, on="CNTRYCODE", how="inner")
    total = total.sort_values(by=["CNTRYCODE"], ascending=[True])
    print(total)


def run_queries(
    root: str,
    storage_options: Dict[str, str],
    queries: List[int],
    gpu: bool,
    dtype_backend: str = None,
):
    """Run TPC-H queries.

    Parameters
    ----------
    root : str
        Path to TPC-H dataset.
    storage_options : Dict[str, str]
        Options for storage connection.
    queries : List[int]
        List of query numbers to run.
    gpu : bool
        Whether to use GPU.
    dtype_backend : str, optional
        Which dtype_backend to use, e.g. whether a DataFrame should use NumPy arrays,
        nullable dtypes or pyarrow for backed data. If None,
        options.dataframe.dtype_backend is used. When "pyarrow" is used,
        columns with supported dtypes are backed by ArrowDtype.
    """
    total_start = time.time()
    print("Start data loading")
    queries_to_args = dict()
    datasets_to_load = set()
    for query in queries:
        args = []
        for dataset in _query_to_datasets[query]:
            args.append(
                globals()[f"load_{dataset}"](
                    root, dtype_backend, gpu, **storage_options
                )
            )
            datasets_to_load.update(args)
        queries_to_args[query] = args
    xorbits.run(list(datasets_to_load))
    print(f"Data loading time (s): {time.time() - total_start}")

    total_start = time.time()
    for query in queries:
        globals()[f"q{query:02}"](*queries_to_args[query])
    print(f"Total query execution time (s): {time.time() - total_start}")


def main():
    parser = argparse.ArgumentParser(description="tpch-queries")
    parser.add_argument(
        "--data_set",
        type=str,
        required=True,
        help="Path to the TPC-H dataset.",
    )
    parser.add_argument(
        "--storage_options",
        type=str,
        required=False,
        help="Path to the storage options json file.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        required=False,
        help="Comma separated TPC-H queries to run.",
    )
    parser.add_argument(
        "--dtype-backend",
        type=str,
        choices=["numpy_nullable", "pyarrow"],
        help="Which dtype_backend to use when reading data",
    )
    parser.add_argument(
        "--gpu",
        default=False,
        action="store_true",
        help="Use GPUs.",
    )
    parser.add_argument(
        "--cuda_devices",
        type=int,
        nargs="+",
        required=False,
        help="Comma separated cuda devices to use.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        required=False,
        help="The endpoint of existing Xorbits cluster.",
    )
    parser.add_argument(
        "--mmap_root_dir",
        type=str,
        required=False,
        help="The directory used by mmap storage backend."
    )

    args = parser.parse_args()
    data_set = args.data_set
    gpu = args.gpu
    mmap_root_dir = args.mmap_root_dir
    if mmap_root_dir is not None:
        print(f"Use mmap to run: {mmap_root_dir}")

    # credentials to access the datasource.
    storage_options = {}
    if args.storage_options is not None:
        with open(args.storage_options, "r") as fp:
            storage_options = json.load(fp)
    print(f"Storage options: {storage_options}")

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")

    if mmap_root_dir is not None:
        xorbits.init(address=args.endpoint, cuda_devices=args.cuda_devices,
                     storage_config={"mmap": {"root_dirs": mmap_root_dir}})
    else:
        xorbits.init(address=args.endpoint, cuda_devices=args.cuda_devices)
    try:
        run_queries(
            data_set,
            storage_options=storage_options,
            queries=queries,
            gpu=gpu,
            dtype_backend=args.dtype_backend,
        )
    finally:
        xorbits.shutdown()


if __name__ == "__main__":
    print(f"Running TPC-H against Xorbits v{xorbits.__version__}")
    main()
