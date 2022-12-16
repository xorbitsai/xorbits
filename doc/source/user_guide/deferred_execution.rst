.. _deferred_execution:

==================
Deferred Execution
==================

.. currentmodule:: xorbits.pandas

Most Xorbits objects, including Xorbits :class:`DataFrame`, are implemented to use deferred
execution. Deferred execution means that operations on Xorbits objects are not executed immediately
as they are called. Instead, Xorbits builds an execution plan and the plan will not be
executed until the result is actually required.

Currently, execution will be triggered in the following situations:

#. ``repr`` is called.
#. ``str`` is called.
#. Output methods are called. For example, :meth:`DataFrame.to_csv` is called.
#. Critical information is missing. For example, the dtypes of a ``DataFrame``.

Deferred execution can greatly improve performance when you manipulate large datasets.
Optimizations can be applied to the chained operations before calling the backend. For example,
identical parts of an execution plan can be eliminated and executed only once.

.. currentmodule:: xorbits

To trigger the execution manually, you can use :func:`run`. You pass an Xorbits object or a list
of Xorbits objects as the argument.

::

    >>> import xorbits.numpy as np
    >>> a = np.arange(3)
    >>> xorbits.run(a)
    >>> a
    array([0, 1, 2])
