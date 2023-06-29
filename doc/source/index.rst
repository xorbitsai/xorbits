.. _index:

.. raw:: html

    <img class="align-center" alt="Xorbits Logo" src="_static/xorbits.svg" style="background-color: transparent", width="77%">

====

Welcome to Xorbits!
"""""""""""""""""""""""""""""""""""""""""

Xorbits is an open-source computing framework that makes it easy to scale ML and DS workloads —
from data preprocessing, to tuning, training, and model serving. Xorbits can leverage multi cores or GPUs to accelerate
computation on a single machine, or scale out up to thousands of machines to support processing terabytes of data.

Xorbits provides a suite of best-in-class libraries for data scientists and machine learning practitioners. Xorbits provides the capability to scale tasks
without the necessity for extensive knowledge of infrastructure.

- :ref:`Xorbits Datasets <xorbits_datasets_index>`: Load and process datasets, from small to large, using the tools you love💜, such as pandas and Numpy.

- :ref:`Xorbits Train <xorbits_train_index>`: Pretrain or finetune your own state-of-the-art models for ML and DL frameworks such as PyTorch, XGBoost, scikit-learn, Hugging Face

- :ref:`Xorbits Serve <xorbits_serve_index>`: Scalable serving to deploy state-of-the-art models. Integrate with the most popular deep learning libraries, like PyTorch, ggml, etc.


Xorbits has familiar Python API which enables pandas, NumPy, scikit-learn, PyTorch, XGBoost, Xarray, mong many others. 
Here is a piece of code that uses Xorbits to scale the pandas library. With just one line of code modification, 
your pandas workflow can be scaled on Xorbits. 


.. image:: _static/pandas_vs_xorbits.gif
   :alt: pandas on Xorbits
   :align: center
   :scale: 70%

As for the name of ``xorbits``, it has many meanings, you can treat it as ``X-or-bits`` or ``X-orbits`` or ``xor-bits``,
just have fun to comprehend it in your own way.


Why Xorbits?
------------

* :ref:`xorbits_vs_pandas`
* :ref:`why_xorbits_fast`
* :ref:`xorbits_vs_dask_modin_koalas`

Getting involved
----------------

.. only:: not zh_cn

    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | **Platform**                                                                                     | **Purpose**                                        |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Discourse Forum <https://discuss.xorbits.io/>`_                                                 | Asking usage questions and discussing development. |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Github Issues <https://github.com/xprobe-inc/xorbits/issues>`_                                  | Reporting bugs and filing feature requests.        |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Slack <https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg>`_  | Collaborating with other Xorbits users.            |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `StackOverflow <https://stackoverflow.com/questions/tagged/xorbits>`_                            | Asking questions about how to use Xorbits.         |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Twitter <https://twitter.com/xorbitsio>`_                                                       | Staying up-to-date on new features.                |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+

.. only:: zh_cn

    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | **Platform**                                                                                     | **Purpose**                                        |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Github Issues <https://github.com/xprobe-inc/xorbits/issues>`_                                  | Reporting bugs and filing feature requests.        |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+
    | `Gitee Issues <https://gitee.com/xprobe-inc/xorbits/issues>`_                                    | Reporting bugs and filing feature requests.        |
    +--------------------------------------------------------------------------------------------------+----------------------------------------------------+

    Visit `Xorbits community <https://xorbits.cn/community>`_ to join us on Wechat, Zhihu and so on.


.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   libraries/index
   user_guide/index
   deployment/index
   reference/index
   development/index
