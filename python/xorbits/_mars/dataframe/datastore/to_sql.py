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

import cloudpickle
import pandas as pd

from ... import opcodes
from ...core import recursive_tile
from ...core.operand import OperatorLogicKeyGeneratorMixin
from ...serialization.serializables import (
    AnyField,
    BoolField,
    BytesField,
    Int64Field,
    StringField,
)
from ..core import DATAFRAME_TYPE
from ..operands import DataFrameOperand, DataFrameOperandMixin
from ..utils import (
    build_empty_df,
    build_empty_series,
    create_sa_connection,
    is_pandas_2,
    parse_index,
)


class DataFrameToSQLTableLogicKeyGeneratorMixin(OperatorLogicKeyGeneratorMixin):
    def _get_logic_key_token_values(self):
        fields_to_tokenize = [
            getattr(self, k, None)
            for k in [
                "table_name",
                "schema",
                "if_exists",
                "index",
                "index_label",
                "chunksize",
                "dtype",
                "method",
            ]
        ]
        return super()._get_logic_key_token_values() + fields_to_tokenize


class DataFrameToSQLTable(
    DataFrameOperand, DataFrameOperandMixin, DataFrameToSQLTableLogicKeyGeneratorMixin
):
    _op_type_ = opcodes.TO_SQL

    table_name = StringField("table_name")
    con = AnyField("con")
    schema = StringField("schema")
    if_exists = StringField("if_exists")
    index = BoolField("index")
    index_label = AnyField("index_label")
    chunksize = Int64Field("chunksize")
    dtype = AnyField("dtype")
    method = AnyField("method")
    engine_kwargs = BytesField(
        "engine_kwargs",
        on_serialize=cloudpickle.dumps,
        on_deserialize=cloudpickle.loads,
        default=None,
    )

    def __call__(self, df_or_series):
        with create_sa_connection(self.con, **(self.engine_kwargs or dict())) as con:
            self.con = str(con.engine.url)
            empty_index = df_or_series.index_value.to_pandas()[:0]
            if isinstance(df_or_series, DATAFRAME_TYPE):
                empty_obj = build_empty_df(df_or_series.dtypes, index=empty_index)
            else:
                empty_obj = build_empty_series(
                    df_or_series.dtype, index=empty_index, name=df_or_series.name
                )

            empty_obj.to_sql(
                self.table_name,
                con=con,
                schema=self.schema,
                if_exists=self.if_exists,
                index=self.index,
                index_label=self.index_label,
                dtype=self.dtype,
            )

            if isinstance(df_or_series, DATAFRAME_TYPE):
                if is_pandas_2():
                    dtypes = pd.Series([], index=pd.RangeIndex(0), dtype="object")
                else:
                    dtypes = df_or_series.dtypes[:0]
                index_value = parse_index(dtypes.index, df_or_series.key, "index")
                columns_value = parse_index(
                    dtypes.index,
                    df_or_series.key,
                    "columns",
                    store_data=True,
                )
                return self.new_dataframe(
                    [df_or_series],
                    shape=(0, 0),
                    dtypes=dtypes,
                    index_value=index_value,
                    columns_value=columns_value,
                )
            else:
                if is_pandas_2():
                    index_value = parse_index(
                        pd.RangeIndex(0), df_or_series.key, "index"
                    )
                else:
                    index_value = parse_index(
                        df_or_series.index_value.to_pandas()[:0],
                        df_or_series.key,
                        "index",
                    )
                return self.new_series(
                    [df_or_series],
                    shape=(0,),
                    dtype=df_or_series.dtype,
                    index_value=index_value,
                )

    @classmethod
    def tile(cls, op: "DataFrameToSQLTable"):
        inp = op.inputs[0]
        out = op.outputs[0]
        if inp.ndim == 2:
            inp = yield from recursive_tile(inp.rechunk({1: (inp.shape[1],)}))

        chunks = []
        for c in inp.chunks:
            new_op = op.copy().reset_key()
            new_op.if_exists = "append"

            index_value = out.index_value
            if c.ndim == 2:
                columns_value = out.columns_value
                chunks.append(
                    new_op.new_chunk(
                        [c],
                        shape=(0, 0),
                        index=c.index,
                        dtypes=out.dtypes,
                        index_value=index_value,
                        columns_value=columns_value,
                    )
                )
            else:
                chunks.append(
                    new_op.new_chunk(
                        [c],
                        shape=(0,),
                        index=c.index,
                        dtype=out.dtype,
                        index_value=index_value,
                    )
                )

        new_op = op.copy().reset_key()
        params = out.params.copy()
        params["nsplits"] = tuple((0,) * len(sp) for sp in inp.nsplits)
        return new_op.new_tileables([inp], chunks=chunks, **params)

    @classmethod
    def execute(cls, ctx, op: "DataFrameToSQLTable"):
        in_df = op.inputs[0]
        out_df = op.outputs[0]
        in_data = ctx[in_df.key]

        import sqlalchemy as sa

        engine = sa.create_engine(op.con, **(op.engine_kwargs or dict()))

        try:
            with engine.connect() as connection:
                with connection.begin():
                    in_data.to_sql(
                        op.table_name,
                        con=connection,
                        if_exists=op.if_exists,
                        index=op.index,
                        index_label=op.index_label,
                        chunksize=op.chunksize,
                        dtype=op.dtype,
                        method=op.method,
                    )
        finally:
            engine.dispose()

        if in_df.ndim == 2:
            ctx[out_df.key] = pd.DataFrame()
        else:
            ctx[out_df.key] = pd.Series([], dtype=in_data.dtype)


def to_sql(
    df,
    name: str,
    con,
    schema=None,
    if_exists: str = "fail",
    index: bool = True,
    index_label=None,
    chunksize=None,
    dtype=None,
    method=None,
):
    """
    Write records stored in a DataFrame to a SQL database.

    Databases supported by SQLAlchemy [1]_ are supported. Tables can be
    newly created, appended to, or overwritten.

    Parameters
    ----------
    name : str
        Name of SQL table.
    con : sqlalchemy.engine.Engine or sqlite3.Connection
        Using SQLAlchemy makes it possible to use any DB supported by that
        library. Legacy support is provided for sqlite3.Connection objects. The user
        is responsible for engine disposal and connection closure for the SQLAlchemy
        connectable See `here                 <https://docs.sqlalchemy.org/en/13/core/connections.html>`_

    schema : str, optional
        Specify the schema (if database flavor supports this). If None, use
        default schema.
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the table already exists.

        * fail: Raise a ValueError.
        * replace: Drop the table before inserting new values.
        * append: Insert new values to the existing table.

    index : bool, default True
        Write DataFrame index as a column. Uses `index_label` as the column
        name in the table.
    index_label : str or sequence, default None
        Column label for index column(s). If None is given (default) and
        `index` is True, then the index names are used.
        A sequence should be given if the DataFrame uses MultiIndex.
    chunksize : int, optional
        Specify the number of rows in each batch to be written at a time.
        By default, all rows will be written at once.
    dtype : dict or scalar, optional
        Specifying the datatype for columns. If a dictionary is used, the
        keys should be the column names and the values should be the
        SQLAlchemy types or strings for the sqlite3 legacy mode. If a
        scalar is provided, it will be applied to all columns.
    method : {None, 'multi', callable}, optional
        Controls the SQL insertion clause used:

        * None : Uses standard SQL ``INSERT`` clause (one per row).
        * 'multi': Pass multiple values in a single ``INSERT`` clause.
        * callable with signature ``(pd_table, conn, keys, data_iter)``.

        Details and a sample callable implementation can be found in the
        section :ref:`insert method <io.sql.method>`.

        .. versionadded:: 0.24.0

    Raises
    ------
    ValueError
        When the table already exists and `if_exists` is 'fail' (the
        default).

    See Also
    --------
    read_sql : Read a DataFrame from a table.

    Notes
    -----
    Timezone aware datetime columns will be written as
    ``Timestamp with timezone`` type with SQLAlchemy if supported by the
    database. Otherwise, the datetimes will be stored as timezone unaware
    timestamps local to the original timezone.

    .. versionadded:: 0.24.0

    References
    ----------
    .. [1] http://docs.sqlalchemy.org
    .. [2] https://www.python.org/dev/peps/pep-0249/

    Examples
    --------

    Create an in-memory SQLite database.

    >>> import mars.dataframe as md
    >>> from sqlalchemy import create_engine
    >>> engine = create_engine('sqlite:////tmp/temp.db')

    Create a table from scratch with 3 rows.

    >>> df = md.DataFrame({'name' : ['User 1', 'User 2', 'User 3']})
    >>> df.execute()
         name
    0  User 1
    1  User 2
    2  User 3

    >>> df.to_sql('users', con=engine).execute()
    >>> engine.execute("SELECT * FROM users").fetchall()
    [(0, 'User 1'), (1, 'User 2'), (2, 'User 3')]

    >>> df1 = md.DataFrame({'name' : ['User 4', 'User 5']})
    >>> df1.to_sql('users', con=engine, if_exists='append').execute()
    >>> engine.execute("SELECT * FROM users").fetchall()
    [(0, 'User 1'), (1, 'User 2'), (2, 'User 3'),
     (0, 'User 4'), (1, 'User 5')]

    Overwrite the table with just ``df1``.

    >>> df1.to_sql('users', con=engine, if_exists='replace',
    ...            index_label='id').execute()
    >>> engine.execute("SELECT * FROM users").fetchall()
    [(0, 'User 4'), (1, 'User 5')]

    Specify the dtype (especially useful for integers with missing values).
    Notice that while pandas is forced to store the data as floating point,
    the database supports nullable integers. When fetching the data with
    Python, we get back integer scalars.

    >>> df = md.DataFrame({"A": [1, None, 2]})
    >>> df.execute()
         A
    0  1.0
    1  NaN
    2  2.0

    >>> from sqlalchemy.types import Integer
    >>> df.to_sql('integers', con=engine, index=False,
    ...           dtype={"A": Integer()}).execute()

    >>> engine.execute("SELECT * FROM integers").fetchall()
    [(1,), (None,), (2,)]
    """
    op = DataFrameToSQLTable(
        table_name=name,
        con=con,
        schema=schema,
        if_exists=if_exists,
        index=index,
        index_label=index_label,
        chunksize=chunksize,
        dtype=dtype,
        method=method,
    )
    return op(df)
