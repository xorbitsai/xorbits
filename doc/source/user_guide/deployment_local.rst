.. _deployment_local:

================
Local deployment
================

Local deployment means that you can run Xorbits on your local machine, e.g. laptop.

Installation
------------

First, ensure Xorbits is correctly installed, if not, see :ref:`installation document <installation>`.

Init Xorbits runtime optionally
-------------------------------

Secondly, you don't have to, but you can optionally init an Xorbits runtime manually.

.. code-block:: python

    import xorbits
    xorbits.init()

Or Xorbits will try to init for you in the background when the first time some computation is triggered.

.. note::

    Initialization of Xorbits may take a while to accomplish, if you try to measure performance of your code
    and so forth, you'd better init Xorbits runtime in advance, or the time of init may be accounted into
    execution time which may be distracted.
