.. _10min_datasets:

======================================
10 minutes to :code:`xorbits.datasets`
======================================

.. currentmodule:: xorbits.datasets

This is a short introduction to :code:`xorbits.datasets`.


Datasets Creation
-----------------

You can create a dataset from Hugging Face datasets, just like the `datasets.load_dataset`. But, 
our dataset loader will load the dataset in parallel*. (Currently, not all the datasets are loaded
in parallel.)

::
    >>> import xorbits.datasets as xdatasets
    >>> dataset = xdatasets.from_huggingface("rotten_tomatoes", split="train")
    Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })

Datasets Processing
-------------------

