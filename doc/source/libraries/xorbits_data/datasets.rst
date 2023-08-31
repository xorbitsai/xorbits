.. _10min_datasets:

======================================
10 minutes to :code:`xorbits.datasets`
======================================

.. currentmodule:: xorbits.datasets

This is a short introduction to :code:`xorbits.datasets`.

Datasets Creation
-----------------

You can create a dataset from Hugging Face datasets, just like the `datasets.load_dataset`.
xorbits.datasets will load the dataset in parallel among multiple machines, while Hugging Face
datasets can only load the dataset in parallel on one machine. Currently, only the dataset with
multiple data files can be loaded in parallel, one data file will be one chunk.

::

    >>> import xorbits.datasets as xdatasets
    >>> dataset = xdatasets.from_huggingface(
    >>>     "mariosasko/test_multi_dir_dataset", split="train")
    Dataset({
        features: ['text'],
        num_rows: 2
    })

Datasets Processing
-------------------

You can apply a function to the dataset, xorbits.datasets will parallelize the apply operations.
xorbits.datasets can process data in parallel at the granularity of chunks if the xorbits cluster
has enough resources. If your dataset has too few chunks, then you can use the `rechunk()` to 
improve concurrency.

::

    >>> import xorbits.datasets as xdatasets
    >>> # The `rotten_tomatoes` dataset contains empty data files, then the dataset
    >>> # will be loaded and auto rechunked by the total cpu of xorbits cluster.
    >>> dataset = xdatasets.from_huggingface("rotten_tomatoes", split="train")
    >>> def add_prefix(example):
    >>>     example["text"] = "Xorbits: " + example["text"]
    >>>     return example
    >>> # Multiple processes applying `add_prefix` concurrently.
    >>> dataset = dataset.map(add_prefix)
    >>> dataset
    Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })
    >>> dataset[1:3]["text"]
    ['Xorbits: the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .',
     'Xorbits: effective but too-tepid biopic']


Datasets Outputs
----------------

Xorbits dataset can be easily converted into xorbits dataframe, then you can continue to process
data by xorbits.pandas, of course in parallel.


::

    >>> df = dataset.to_dataframe()
    >>> # You may want to reset the index.
    >>> df.reset_index(drop=True, inplace=True)
    >>> df
                                                    text  label
    0     Xorbits: the rock is destined to be the 21st c...      1
    1     Xorbits: this is a film well worth seeing , ta...      1
    2     Xorbits: a thoughtful , provocative , insisten...      1
    3     Xorbits: guaranteed to move anyone who ever sh...      1
    4     Xorbits: newton draws our attention like a mag...      1
                                                    ...    ...
    8525  Xorbits: a laughable -- or rather , unlaughabl...      0
    8526  Xorbits: plays like an unbalanced mixture of g...      0
    8527  Xorbits: i wish it would have just gone more o...      0
    8528  Xorbits: like its title character , esther kah...      0
    8529  Xorbits: things really get weird , though not ...      0
    [8530 rows x 2 columns]
    >>> # The xorbits.pandas operations are executed in parallel among the cluster.
    >>> df["label"].sum()
    4265
