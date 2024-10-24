Statistics
==========

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

Order statistics
----------------

+-------------------+------------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``        | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+------------------+------------------------+----------------------------------+
| ``ptp``           | `ptp`_           | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``percentile``    | `percentile`_    | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``nanpercentile`` | `nanpercentile`_ | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``quantile``      | `quantile`_      | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``nanquantile``   | `nanquantile`_   | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+

Averages and variances
----------------------

+-------------------+--------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``    | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+--------------+------------------------+----------------------------------+
| ``median``        | `median`_    | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``average``       | `average`_   | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``mean``          | `mean`_      | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``std``           | `std`_       | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``var``           | `var`_       | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``nanmedian``     | `nanmedian`_ | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``nanmean``       | `nanmean`_   | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``nanstd``        | `nanstd`_    | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``nanvar``        | `nanvar`_    | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+

Correlating
-----------

+-------------------+--------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``    | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+--------------+------------------------+----------------------------------+
| ``corrcoef``      | `corrcoef`_  | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``correlate``     | `correlate`_ | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``cov``           | `cov`_       | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+

Histograms
----------

+-------------------------+------------------------+------------------------+----------------------------------+
| ``xorbits.numpy``       | ``numpy``              | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``histogram``           | `histogram`_           | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``histogram2d``         | `histogram2d`_         | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``histogramdd``         | `histogramdd`_         | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``bincount``            | `bincount`_            | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``histogram_bin_edges`` | `histogram_bin_edges`_ | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+
| ``digitize``            | `digitize`_            | Y                      |                                  |
+-------------------------+------------------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`ptp`: https://numpy.org/doc/stable/reference/generated/numpy.ptp.html
.. _`percentile`: https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
.. _`nanpercentile`: https://numpy.org/doc/stable/reference/generated/numpy.nanpercentile.html
.. _`quantile`: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
.. _`nanquantile`: https://numpy.org/doc/stable/reference/generated/numpy.nanquantile.html
.. _`median`: https://numpy.org/doc/stable/reference/generated/numpy.median.html
.. _`average`: https://numpy.org/doc/stable/reference/generated/numpy.average.html
.. _`mean`: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
.. _`std`: https://numpy.org/doc/stable/reference/generated/numpy.std.html
.. _`var`: https://numpy.org/doc/stable/reference/generated/numpy.var.html
.. _`nanmedian`: https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html
.. _`nanmean`: https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
.. _`nanstd`: https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html
.. _`nanvar`: https://numpy.org/doc/stable/reference/generated/numpy.nanvar.html
.. _`corrcoef`: https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
.. _`correlate`: https://numpy.org/doc/stable/reference/generated/numpy.correlate.html
.. _`cov`: https://numpy.org/doc/stable/reference/generated/numpy.cov.html
.. _`histogram`: https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
.. _`histogram2d`: https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
.. _`histogramdd`: https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html
.. _`bincount`: https://numpy.org/doc/stable/reference/generated/numpy.bincount.html
.. _`histogram_bin_edges`: https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
.. _`digitize`: https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
