.. _api.io:

============
Input/output
============

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding pandas method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to pandas.

Pickling
~~~~~~~~

+-------------------------+------------------------+------------------------+----------------------------------+
| ``xorbits.pandas``      | ``pandas``             | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``read_pickle``         | `read_pickle`_         | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``DataFrame.to_pickle`` | `DataFrame.to_pickle`_ | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+

Flat file
~~~~~~~~~

+----------------------+---------------------+------------------------+----------------------------------+
| ``xorbits.pandas``   | ``pandas``          | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------+---------------------+------------------------+----------------------------------+
| ``read_table``       | `read_table`_       | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``read_csv``         | `read_csv`_         | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``DataFrame.to_csv`` | `DataFrame.to_csv`_ | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``read_fwf``         | `read_fwf`_         | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+

Clipboard
~~~~~~~~~

+----------------------------+---------------------------+------------------------+----------------------------------+
| ``xorbits.pandas``         | ``pandas``                | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------------+---------------------------+------------------------+----------------------------------+
| ``read_clipboard``         | `read_clipboard`_         | Y                      |                                  |
+----------------------------+---------------------------+------------------------+----------------------------------+
| ``DataFrame.to_clipboard`` | `DataFrame.to_clipboard`_ | Y                      |                                  |
+----------------------------+---------------------------+------------------------+----------------------------------+

Excel
~~~~~

+------------------------+-----------------------+------------------------+----------------------------------+
| ``xorbits.pandas``     | ``pandas``            | Implemented? (Y/N/P/D) | Notes for Current implementation |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``read_excel``         | `read_excel`_         | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``DataFrame.to_excel`` | `DataFrame.to_excel`_ | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``ExcelFile.parse``    | `ExcelFile.parse`_    | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``ExcelWriter``        | `ExcelWriter`_        | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+

+-------------------------------------+-----------------------------+------------------------+----------------------------------+
| ``xorbits.pandas.io.formats.style`` | ``pandas.io.formats.style`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------------------------+-----------------------------+------------------------+----------------------------------+
| ``Styler.to_excel``                 | `Styler.to_excel`_          | Y                      |                                  |
+-------------------------------------+-----------------------------+------------------------+----------------------------------+

JSON
~~~~
+-----------------------+----------------------+------------------------+----------------------------------+
| ``xorbits.pandas``    | ``pandas``           | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``read_json``         | `read_json`_         | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``json_normalize``    | `json_normalize`_    | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``DataFrame.to_json`` | `DataFrame.to_json`_ | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+

+----------------------------+-----------------------+------------------------+----------------------------------+
| ``xorbits.pandas.io.json`` | ``pandas.io.json``    | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------------+-----------------------+------------------------+----------------------------------+
| ``build_table_schema``     | `build_table_schema`_ | Y                      |                                  |
+----------------------------+-----------------------+------------------------+----------------------------------+

HTML
~~~~

+-----------------------+----------------------+------------------------+----------------------------------+
| ``xorbits.pandas``    | ``pandas``           | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``read_html``         | `read_html`_         | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+
| ``DataFrame.to_html`` | `DataFrame.to_html`_ | Y                      |                                  |
+-----------------------+----------------------+------------------------+----------------------------------+

+--------------------+-------------------+------------------------+----------------------------------+
| ``xorbits.pandas`` | ``pandas``        | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+-------------------+------------------------+----------------------------------+
| ``Styler.to_html`` | `Styler.to_html`_ | Y                      |                                  |
+--------------------+-------------------+------------------------+----------------------------------+

XML
~~~~

+----------------------+---------------------+------------------------+----------------------------------+
| ``xorbits.pandas``   | ``pandas``          | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------+---------------------+------------------------+----------------------------------+
| ``read_xml``         | `read_xml`_         | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``DataFrame.to_xml`` | `DataFrame.to_xml`_ | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+

Latex
~~~~~

+------------------------+-----------------------+------------------------+----------------------------------+
| ``xorbits.pandas``     | ``pandas``            | Implemented? (Y/N/P/D) | Notes for Current implementation |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``DataFrame.to_latex`` | `DataFrame.to_latex`_ | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+

+-------------------------------------+-----------------------------+------------------------+----------------------------------+
| ``xorbits.pandas.io.formats.style`` | ``pandas.io.formats.style`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------------------------+-----------------------------+------------------------+----------------------------------+
| ``Styler.to_latex``                 | `Styler.to_latex`_          | Y                      |                                  |
+-------------------------------------+-----------------------------+------------------------+----------------------------------+

HDFStore: PyTables (HDF5)
~~~~~~~~~~~~~~~~~~~~~~~~~

+---------------------+--------------------+------------------------+----------------------------------+
| ``xorbits.pandas``  | ``pandas``         | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------+--------------------+------------------------+----------------------------------+
| ``read_hdf``        | `read_hdf`_        | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``HDFStore.put``    | `HDFStore.put`_    | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``HDFStore.append`` | `HDFStore.append`_ | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``HDFStore.get``    | `HDFStore.get`_    | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``HDFStore.select`` | `HDFStore.select`_ | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``HDFStore.info``   | `HDFStore.info`_   | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``HDFStore.keys``   | `HDFStore.keys`_   | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``HDFStore.groups`` | `HDFStore.groups`_ | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+
| ``HDFStore.walk``   | `HDFStore.walk`_   | Y                      |                                  |
+---------------------+--------------------+------------------------+----------------------------------+

.. warning::

   One can store a subclass of :class:`DataFrame` or :class:`Series` to HDF5,
   but the type of the subclass is lost upon storing.

Feather
~~~~~~~

+--------------------------+-------------------------+------------------------+----------------------------------+
| ``xorbits.pandas``       | ``pandas``              | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``read_feather``         | `read_feather`_         | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``DataFrame.to_feather`` | `DataFrame.to_feather`_ | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+

Parquet
~~~~~~~

+--------------------------+-------------------------+------------------------+----------------------------------+
| ``xorbits.pandas``       | ``pandas``              | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``read_parquet``         | `read_parquet`_         | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``DataFrame.to_parquet`` | `DataFrame.to_parquet`_ | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+

ORC
~~~

+----------------------+---------------------+------------------------+----------------------------------+
| ``xorbits.pandas``   | ``pandas``          | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------+---------------------+------------------------+----------------------------------+
| ``read_orc``         | `read_orc`_         | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``DataFrame.to_orc`` | `DataFrame.to_orc`_ | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+

SAS
~~~

+--------------------+-------------+------------------------+----------------------------------+
| ``xorbits.pandas`` | ``pandas``  | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+-------------+------------------------+----------------------------------+
| ``read_sas``       | `read_sas`_ | Y                      |                                  |
+--------------------+-------------+------------------------+----------------------------------+

SPSS
~~~~

+--------------------+--------------+------------------------+----------------------------------+
| ``xorbits.pandas`` | ``pandas``   | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+--------------+------------------------+----------------------------------+
| ``read_spss``      | `read_spss`_ | Y                      |                                  |
+--------------------+--------------+------------------------+----------------------------------+

SQL
~~~

+----------------------+---------------------+------------------------+----------------------------------+
| ``xorbits.pandas``   | ``pandas``          | Implemented? (Y/N/P/D) | Notes for Current implementation |
+----------------------+---------------------+------------------------+----------------------------------+
| ``read_sql_table``   | `read_sql_table`_   | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``read_sql_query``   | `read_sql_query`_   | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``read_sql``         | `read_sql`_         | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+
| ``DataFrame.to_sql`` | `DataFrame.to_sql`_ | Y                      |                                  |
+----------------------+---------------------+------------------------+----------------------------------+

Google BigQuery
~~~~~~~~~~~~~~~

+--------------------+-------------+------------------------+----------------------------------+
| ``xorbits.pandas`` | ``pandas``  | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------+-------------+------------------------+----------------------------------+
| ``read_gbq``       | `read_gbq`_ | Y                      |                                  |
+--------------------+-------------+------------------------+----------------------------------+

STATA
~~~~~

+------------------------+-----------------------+------------------------+----------------------------------+
| ``xorbits.pandas``     | ``pandas``            | Implemented? (Y/N/P/D) | Notes for Current implementation |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``read_stata``         | `read_stata`_         | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+
| ``DataFrame.to_stata`` | `DataFrame.to_stata`_ | Y                      |                                  |
+------------------------+-----------------------+------------------------+----------------------------------+


+---------------------------------+--------------------------------+------------------------+----------------------------------+
| ``xorbits.pandas.io.stata``     | ``pandas.io.stata``            | Implemented? (Y/N/P/D) | Notes for Current implementation |
+---------------------------------+--------------------------------+------------------------+----------------------------------+
| ``StataReader.data_label``      | `StataReader.data_label`_      | Y                      |                                  |
+---------------------------------+--------------------------------+------------------------+----------------------------------+
| ``StataReader.value_labels``    | `StataReader.value_labels`_    | Y                      |                                  |
+---------------------------------+--------------------------------+------------------------+----------------------------------+
| ``StataReader.variable_labels`` | `StataReader.variable_labels`_ | Y                      |                                  |
+---------------------------------+--------------------------------+------------------------+----------------------------------+
| ``StataWriter.write_file``      | `StataWriter.write_file`_      | Y                      |                                  |
+---------------------------------+--------------------------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`read_pickle`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html
.. _`DataFrame.to_pickle`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_pickle.html
.. _`read_table`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_table.html
.. _`read_csv`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
.. _`DataFrame.to_csv`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
.. _`read_fwf`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_fwf.html
.. _`read_clipboard`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_clipboard.html
.. _`DataFrame.to_clipboard`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_clipboard.html
.. _`read_excel`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html
.. _`DataFrame.to_excel`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_excel.html
.. _`ExcelFile.parse`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.ExcelFile.parse.html
.. _`Styler.to_excel`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.to_excel.html
.. _`ExcelWriter`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.ExcelWriter.html
.. _`read_json`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html
.. _`json_normalize`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html
.. _`DataFrame.to_json`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html
.. _`build_table_schema`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.json.build_table_schema.html
.. _`read_html`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_html.html
.. _`DataFrame.to_html`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_html.html
.. _`Styler.to_html`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.to_html.html
.. _`read_xml`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_xml.html
.. _`DataFrame.to_xml`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_xml.html
.. _`DataFrame.to_latex`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_latex.html
.. _`Styler.to_latex`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.to_latex.html
.. _`read_hdf`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html
.. _`HDFStore.put`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.put.html
.. _`HDFStore.append`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.append.html
.. _`HDFStore.get`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.get.html
.. _`HDFStore.select`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.select.html
.. _`HDFStore.info`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.info.html
.. _`HDFStore.keys`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.keys.html
.. _`HDFStore.groups`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.groups.html
.. _`HDFStore.walk`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.HDFStore.walk.html
.. _`read_feather`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_feather.html
.. _`DataFrame.to_feather`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_feather.html
.. _`read_parquet`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html
.. _`DataFrame.to_parquet`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_parquet.html
.. _`read_orc`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_orc.html
.. _`DataFrame.to_orc`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_orc.html
.. _`read_sas`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sas.html
.. _`read_spss`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_spss.html
.. _`read_sql_table`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql_table.html
.. _`read_sql_query`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql_query.html
.. _`read_sql`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html
.. _`DataFrame.to_sql`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html
.. _`read_gbq`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_gbq.html
.. _`read_stata`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_stata.html
.. _`DataFrame.to_stata`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_stata.html
.. _`StataReader.data_label`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.stata.StataReader.data_label.html
.. _`StataReader.value_labels`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.stata.StataReader.value_labels.html
.. _`StataReader.variable_labels`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.stata.StataReader.variable_labels.html
.. _`StataWriter.write_file`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.StataWriter.write_file.html
