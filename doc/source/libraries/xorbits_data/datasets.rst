.. _10min_datasets:

===================================
10 minutes to :code:`xorbits.datasets`
===================================

.. currentmodule:: xorbits.datasets

This is a short introduction to :code:`xorbits.datasets`.


Datasets Creation
-----------------

::
    import xorbits.datasets as xds

    dataset = xds.from_huggingface("rotten_tomatoes", split="train")