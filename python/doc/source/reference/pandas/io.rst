.. _api.io:

============
Input/output
============
.. currentmodule:: xorbits.pandas

Pickling
~~~~~~~~
.. autosummary::
   :toctree: api/

   read_pickle
   DataFrame.to_pickle

Flat file
~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_table
   read_csv
   DataFrame.to_csv
   read_fwf

Clipboard
~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_clipboard
   DataFrame.to_clipboard

Excel
~~~~~
.. autosummary::
   :toctree: api/

   read_excel
   DataFrame.to_excel
   ExcelFile.parse

.. currentmodule:: xorbits.pandas.io.formats.style

.. autosummary::
   :toctree: api/

   Styler.to_excel

.. currentmodule:: xorbits.pandas

.. autosummary::
   :toctree: api/

   ExcelWriter

.. currentmodule:: xorbits.pandas

JSON
~~~~
.. autosummary::
   :toctree: api/

   read_json
   json_normalize
   DataFrame.to_json

.. currentmodule:: xorbits.pandas.io.json

.. autosummary::
   :toctree: api/

   build_table_schema

.. currentmodule:: xorbits.pandas

HTML
~~~~
.. autosummary::
   :toctree: api/

   read_html
   DataFrame.to_html

.. currentmodule:: xorbits.pandas.io.formats.style

.. autosummary::
   :toctree: api/

   Styler.to_html

.. currentmodule:: xorbits.pandas

XML
~~~~
.. autosummary::
   :toctree: api/

   read_xml
   DataFrame.to_xml

Latex
~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.to_latex

.. currentmodule:: xorbits.pandas.io.formats.style

.. autosummary::
   :toctree: api/

   Styler.to_latex

.. currentmodule:: xorbits.pandas

HDFStore: PyTables (HDF5)
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_hdf
   HDFStore.put
   HDFStore.append
   HDFStore.get
   HDFStore.select
   HDFStore.info
   HDFStore.keys
   HDFStore.groups
   HDFStore.walk

.. warning::

   One can store a subclass of :class:`DataFrame` or :class:`Series` to HDF5,
   but the type of the subclass is lost upon storing.

Feather
~~~~~~~
.. autosummary::
   :toctree: api/

   read_feather
   DataFrame.to_feather

Parquet
~~~~~~~
.. autosummary::
   :toctree: api/

   read_parquet
   DataFrame.to_parquet

ORC
~~~
.. autosummary::
   :toctree: api/

   read_orc
   DataFrame.to_orc

SAS
~~~
.. autosummary::
   :toctree: api/

   read_sas

SPSS
~~~~
.. autosummary::
   :toctree: api/

   read_spss

SQL
~~~
.. autosummary::
   :toctree: api/

   read_sql_table
   read_sql_query
   read_sql
   DataFrame.to_sql

Google BigQuery
~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   read_gbq

STATA
~~~~~
.. autosummary::
   :toctree: api/

   read_stata
   DataFrame.to_stata

.. currentmodule:: xorbits.pandas.io.stata

.. autosummary::
   :toctree: api/

   StataReader.data_label
   StataReader.value_labels
   StataReader.variable_labels
   StataWriter.write_file
