.. _api.io:

============
Input/output
============
.. currentmodule:: xorbits.pandas

Pickling
~~~~~~~~
.. autosummary::
   :toctree: generated/

   read_pickle
   DataFrame.to_pickle

Flat file
~~~~~~~~~
.. autosummary::
   :toctree: generated/

   read_table
   read_csv
   DataFrame.to_csv
   read_fwf

Clipboard
~~~~~~~~~
.. autosummary::
   :toctree: generated/

   read_clipboard
   DataFrame.to_clipboard

Excel
~~~~~
.. autosummary::
   :toctree: generated/

   read_excel
   DataFrame.to_excel
   ExcelFile.parse

.. currentmodule:: xorbits.pandas.io.formats.style

.. autosummary::
   :toctree: generated/

   Styler.to_excel

.. currentmodule:: xorbits.pandas

.. autosummary::
   :toctree: generated/

   ExcelWriter

.. currentmodule:: xorbits.pandas

JSON
~~~~
.. autosummary::
   :toctree: generated/

   read_json
   json_normalize
   DataFrame.to_json

.. currentmodule:: xorbits.pandas.io.json

.. autosummary::
   :toctree: generated/

   build_table_schema

.. currentmodule:: xorbits.pandas

HTML
~~~~
.. autosummary::
   :toctree: generated/

   read_html
   DataFrame.to_html

.. currentmodule:: xorbits.pandas.io.formats.style

.. autosummary::
   :toctree: generated/

   Styler.to_html

.. currentmodule:: xorbits.pandas

XML
~~~~
.. autosummary::
   :toctree: generated/

   read_xml
   DataFrame.to_xml

Latex
~~~~~
.. autosummary::
   :toctree: generated/

   DataFrame.to_latex

.. currentmodule:: xorbits.pandas.io.formats.style

.. autosummary::
   :toctree: generated/

   Styler.to_latex

.. currentmodule:: xorbits.pandas

HDFStore: PyTables (HDF5)
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

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
   :toctree: generated/

   read_feather
   DataFrame.to_feather

Parquet
~~~~~~~
.. autosummary::
   :toctree: generated/

   read_parquet
   DataFrame.to_parquet

ORC
~~~
.. autosummary::
   :toctree: generated/

   read_orc
   DataFrame.to_orc

SAS
~~~
.. autosummary::
   :toctree: generated/

   read_sas

SPSS
~~~~
.. autosummary::
   :toctree: generated/

   read_spss

SQL
~~~
.. autosummary::
   :toctree: generated/

   read_sql_table
   read_sql_query
   read_sql
   DataFrame.to_sql

Google BigQuery
~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   read_gbq

STATA
~~~~~
.. autosummary::
   :toctree: generated/

   read_stata
   DataFrame.to_stata

.. currentmodule:: xorbits.pandas.io.stata

.. autosummary::
   :toctree: generated/

   StataReader.data_label
   StataReader.value_labels
   StataReader.variable_labels
   StataWriter.write_file
