.. _loading_data:

==============
Loading data
==============

Xorbits supports reading data from various data sources, including csv, parquet, sql, xml and other data formats,
but not every data format supports parallel reading. We recommend using formats that support parallel reading,
including:
- :func:`xorbits.pandas.read_parquet`
- :func:`xorbits.pandas.read_csv`
- :func:`xorbits.pandas.read_sql_table`
- :func:`xorbits.pandas.read_sql_query`
- :func:`xorbits.pandas.read_sql`

For reading Parquet files, followings are some best practices.

Parquet files
--------------

Parquet is a very popular columnar storage file format. Xorbits Pandas supports parallel
reading and writing of Parquet files. Here we will introduce how to use these functions and
some best practice recommendations.

:func:`xorbits.pandas.read_parquet` accepts the following input forms:

- Single Parquet file
- Folder contains parquet files
- String with wildcards

All these can be local files or remote storage.

.. code-block:: python

    import xorbits.pandas as pd

    # single local file
    df = pd.read_parquet("local.parquet")

    # S3 directory
    df = pd.read_parquet("s3://bucket-name/parquet/files",
                         storage_options={"key": "", "secret": ""})

    # wildcard in path
    df = pd.read_parquet("s3://bucket-name/*.parquet",
                         storage_options={"key": "", "secret": ""})


Store as multiple files
^^^^^^^^^^^^^^^^^^^^^^^^
When the data is large, for best performance, it is best for users to store using multiple Parquet files.
Xorbits will utilize multiprocessing or distributed workers to read different files in parallel to accelerate reading.
Each file will become a Xorbits chunk, and more chunks allow higher concurrency. Generally recommending each
Parquet file to be 16MiB ~ 128MiB in size, so there are not too many files but concurrency can be guaranteed.

For example with 200MiB of data, single file:

.. code-block:: python

    In [1]: %time print(pd.read_parquet("single.parquet"))
    100%|████████████████████████████████████| 100.00/100 [00:01<00:00, 80.31it/s]
                 col1                                               col2     col3
    0        0.804201  Surface play great information. Make enjoy vot...        0
    1        0.602314  Pattern arrive image everyone manager. Traditi...        1
    2        0.416683  Mention central gun especially fish family. He...        2
    3        0.697665  Congress others become that. Life reveal gener...        3
    4        0.774197  Wife though bring inside industry drug. Unit w...        4
    ...           ...                                                ...      ...
    1999995  0.123357  Through child behavior scene. Character simply...  1999995
    1999996  0.983500  Admit laugh peace west recently why free few. ...  1999996
    1999997  0.341014  Class necessary event radio material nearly im...  1999997
    1999998  0.790413  Operation that interesting summer a identify. ...  1999998
    1999999  0.553956  Take receive future situation. Per industry ki...  1999999

    [2000000 rows x 3 columns]
    CPU times: user 402 ms, sys: 165 ms, total: 567 ms
    Wall time: 1.81 s

Stored the same data in a folder with 10 Parquet files, reading the folder:

.. code-block:: python

    In [2]: %time print(pd.read_parquet("parquet_dir"))
    100%|████████████████████████████████████| 100.00/100 [00:00<00:00, 419.56it/s]
                 col1                                               col2     col3
    0        0.804201  Surface play great information. Make enjoy vot...        0
    1        0.602314  Pattern arrive image everyone manager. Traditi...        1
    2        0.416683  Mention central gun especially fish family. He...        2
    3        0.697665  Congress others become that. Life reveal gener...        3
    4        0.774197  Wife though bring inside industry drug. Unit w...        4
    ...           ...                                                ...      ...
    1999995  0.123357  Through child behavior scene. Character simply...  1999995
    1999996  0.983500  Admit laugh peace west recently why free few. ...  1999996
    1999997  0.341014  Class necessary event radio material nearly im...  1999997
    1999998  0.790413  Operation that interesting summer a identify. ...  1999998
    1999999  0.553956  Take receive future situation. Per industry ki...  1999999

    [2000000 rows x 3 columns]
    CPU times: user 117 ms, sys: 30.3 ms, total: 147 ms
    Wall time: 302 ms

From the running time we can see reading multiple files takes only 1/6 the time of a single file.

Single Parquet file with multiple row groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If storing as a single file, splitting into multiple row groups can also allow parallel reading. First use the
`row_group_size` parameter to store into multiple row groups.

.. code-block:: python

    In [3]: df.to_parquet("all.parquet", row_group_size=20_0000)

When reading, specify `groups_as_chunks=True`:

.. code-block:: python

    In [4]: %time print(pd.read_parquet("all.parquet", groups_as_chunks=True))
    100%|███████████████████████████████████| 100.00/100 [00:00<00:00, 231.36it/s]
                col1                                               col2     col3
    0       0.804201  Surface play great information. Make enjoy vot...        0
    1       0.602314  Pattern arrive image everyone manager. Traditi...        1
    2       0.416683  Mention central gun especially fish family. He...        2
    3       0.697665  Congress others become that. Life reveal gener...        3
    4       0.774197  Wife though bring inside industry drug. Unit w...        4
    ...          ...                                                ...      ...
    199995  0.123357  Through child behavior scene. Character simply...  1999995
    199996  0.983500  Admit laugh peace west recently why free few. ...  1999996
    199997  0.341014  Class necessary event radio material nearly im...  1999997
    199998  0.790413  Operation that interesting summer a identify. ...  1999998
    199999  0.553956  Take receive future situation. Per industry ki...  1999999

    [2000000 rows x 3 columns]
    CPU times: user 108 ms, sys: 39.5 ms, total: 147 ms
    Wall time: 508 ms

Acceleration can also be achieved.


Use `rebalance` to redistribute data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If unable to modify the data source, having just a single file will cause data skew problems in following
computations. In this case, call `df.rebalance` after reading Parquet to evenly distribute the data to each worker
and process.

Reading a single Parquet file and calling apply function then, this does not leverage multi-core parallelism:

.. code-block:: python

    In [5]: %time print(pd.read_parquet("all.parquet").apply(lambda row: len(row[1]) * row[2], axis=1))
    100%|███████████████████████████████████| 100.00/100 [00:06<00:00, 16.10it/s]
    0                  0
    1                117
    2                312
    3                519
    4                780
                 ...
    1999995    205999485
    1999996    219999560
    1999997    373999439
    1999998    397999602
    1999999    369999815
    Length: 2000000, dtype: int64
    CPU times: user 39.9 ms, sys: 11.5 ms, total: 51.4 ms
    Wall time: 6.22 s

Upon calling rebalance, the computation will make use of multiple cores, although `rebalance` will consume
some additional time, the more subsequent computations, the higher the gain.

.. code-block:: python

    In [6]: %time print(pd.read_parquet("all.parquet").rebalance().apply(lambda row: len(row[1]) * row[2], axis=1))
    100%|███████████████████████████████████| 100.00/100 [00:04<00:00, 20.16it/s]
    0                  0
    1                117
    2                312
    3                519
    4                780
                 ...
    1999995    205999485
    1999996    219999560
    1999997    373999439
    1999998    397999602
    1999999    369999815
    Length: 2000000, dtype: int64
    CPU times: user 163 ms, sys: 46.9 ms, total: 210 ms
    Wall time: 4.98 s

After repartitioning data, the computational acceleration of apply saved 20% of the computing time for
the whole calculation.
