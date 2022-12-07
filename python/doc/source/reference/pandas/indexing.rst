.. _api.indexing:

=============
Index objects
=============

Index
-----
.. currentmodule:: xorbits.pandas

**Many of these methods or variants thereof are available on the objects
that contain an index (Series/DataFrame) and those should most likely be
used before calling these methods directly.**

.. autosummary::
   :toctree: api/

   Index

Properties
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.values
   Index.is_monotonic
   Index.is_monotonic_increasing
   Index.is_monotonic_decreasing
   Index.is_unique
   Index.has_duplicates
   Index.hasnans
   Index.dtype
   Index.inferred_type
   Index.is_all_dates
   Index.shape
   Index.name
   Index.names
   Index.nbytes
   Index.ndim
   Index.size
   Index.empty
   Index.T
   Index.memory_usage

Modifying and computations
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.all
   Index.any
   Index.argmin
   Index.argmax
   Index.copy
   Index.delete
   Index.drop
   Index.drop_duplicates
   Index.duplicated
   Index.equals
   Index.factorize
   Index.identical
   Index.insert
   Index.is_
   Index.is_boolean
   Index.is_categorical
   Index.is_floating
   Index.is_integer
   Index.is_interval
   Index.is_mixed
   Index.is_numeric
   Index.is_object
   Index.min
   Index.max
   Index.reindex
   Index.rename
   Index.repeat
   Index.where
   Index.take
   Index.putmask
   Index.unique
   Index.nunique
   Index.value_counts

Compatibility with MultiIndex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.set_names
   Index.droplevel

Missing values
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.fillna
   Index.dropna
   Index.isna
   Index.notna

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.astype
   Index.item
   Index.map
   Index.ravel
   Index.to_list
   Index.to_native_types
   Index.to_series
   Index.to_frame
   Index.view
