.. _10min_datasets:

======================================
10 minutes to :code:`xorbits.datasets`
======================================

.. currentmodule:: xorbits.datasets

This is a short introduction to :code:`xorbits.datasets`.

Datasets Creation
-----------------

You can create a dataset from Hugging Face datasets, just like the `datasets.load_dataset`. But, 
xorbits.datasets will load the dataset in parallel*. *(Currently, not all the datasets are loaded
in parallel.)*

::

    >>> import xorbits.datasets as xdatasets
    >>> dataset = xdatasets.from_huggingface("rotten_tomatoes", split="train")
    Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })

Datasets Processing
-------------------

You can apply a function to the dataset, xorbits.datasets will parallel the apply operations.
Dataset can process data in parallel at the granularity of chunks if the xorbits cluster has
enough resources.

::

    >>> def add_prefix(example):
    >>>     example["text"] = "Xorbits: " + example["text"]
    >>>     return example
    >>> dataset = dataset.map(add_prefix)
    >>> # Currently, you have to execute() and fetch() to get all the dataset.
    >>> dataset.execute()
    >>> dataset.fetch()

Datasets Outputs
---------------

Xorbits dataset can be easily converted into xorbits dataframe, then you can continue to process
data by xorbits.pandas.


::

    >>> df = dataset.to_dataframe()
                                                    text  label
    0     the rock is destined to be the 21st century's ...      1
    1     the gorgeously elaborate continuation of " the...      1
    2                        effective but too-tepid biopic      1
    3     if you sometimes like to go to the movies to h...      1
    4     emerges as something rare , an issue movie tha...      1
                                                    ...    ...
    8525  any enjoyment will be hinge from a personal th...      0
    8526  if legendary shlockmeister ed wood had ever ma...      0
    8527  hardly a nuanced portrait of a young woman's b...      0
    8528    interminably bleak , to say nothing of boring .      0
    8529  things really get weird , though not particula...      0
    [8530 rows x 2 columns]
    >>> df["label"].sum()
    4265
