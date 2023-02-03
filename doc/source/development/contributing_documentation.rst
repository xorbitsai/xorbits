.. _contributing_documentation:

=================================
Contributing to the documentation
=================================

Contributing to the documentation benefits everyone who uses Xorbits.
We encourage you to help us improve the documentation, and
you don't have to be an expert on Xorbits to do so! In fact,
there are sections of the docs that are worse off after being written by
experts. If something in the docs doesn't make sense to you, updating the
relevant section after you figure it out is a great way to ensure it will help
the next person. Please visit the `issues page <https://github.com/xprobe-inc/xorbits/issues?page=1&q=is%3Aopen+sort%3Aupdated-desc+label%3ADocs>`__
for a full list of issues that are currently open regarding the
Xorbits documentation.



.. contents:: Documentation:
   :local:


About the Xorbits documentation
-------------------------------

The documentation is written in **reStructuredText**, which is almost like writing
in plain English, and built using `Sphinx <https://www.sphinx-doc.org/en/master/>`__. The
Sphinx Documentation has an excellent `introduction to reST
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`__. Review the Sphinx docs to perform more
complex changes to the documentation as well.

Some other important things to know about the docs:

* The Xorbits documentation consists of two parts: the docstrings in the code
  itself and the docs in this folder ``doc/``.

  The docstrings provide a clear explanation of the usage of the individual
  functions, while the documentation in this folder consists of tutorial-like
  overviews per topic together with some other information (what's new,
  installation, etc).

* The docstrings follow a Xorbits convention, based on the **Numpy Docstring
  Standard**.

* The tutorials make heavy use of the `IPython directive
  <https://matplotlib.org/sampledoc/ipython_directive.html>`_ sphinx extension.
  This directive lets you put code in the documentation which will be run
  during the doc build. For example::

      .. ipython:: python

          x = 2
          x**3

  will be rendered as::

      In [1]: x = 2

      In [2]: x**3
      Out[2]: 8

  Almost all code examples in the docs are run (and the output saved) during the
  doc build. This approach means that code examples will always be up to date,
  but it does make the doc building a bit more complex.

* Our API documentation files in ``doc/source/reference`` house the auto-generated
  documentation from the docstrings. For classes, there are a few subtleties
  around controlling which methods and attributes have pages auto-generated.

  We have two autosummary templates for classes.

  1. ``_templates/autosummary/class.rst``. Use this when you want to
     automatically generate a page for every public method and attribute on the
     class. The ``Attributes`` and ``Methods`` sections will be automatically
     added to the class' rendered documentation by numpydoc. See ``DataFrame``
     for an example.

  2. ``_templates/autosummary/class_without_autosummary``. Use this when you
     want to pick a subset of methods / attributes to auto-generate pages for.
     When using this template, you should include an ``Attributes`` and
     ``Methods`` section in the class docstring. See ``CategoricalIndex`` for an
     example.

  Every method should be included in a ``toctree`` in one of the documentation files in
  ``doc/source/reference``, else Sphinx
  will emit a warning.

The utility script ``scripts/validate_docstrings.py`` can be used to get a csv
summary of the API documentation. And also validate common errors in the docstring
of a specific class, function or method. The summary also compares the list of
methods documented in the files in ``doc/source/reference`` (which is used to generate
the `API Reference <https://doc.xorbits.io/en/latest/reference/index.html>`_ page)
and the actual public methods.
This will identify methods documented in ``doc/source/reference`` that are not actually
class methods, and existing methods that are not documented in ``doc/source/reference``.


Updating a Xorbits docstring
----------------------------

When improving a single function or method's docstring, it is not necessarily
needed to build the full documentation (see next section).

The examples in the docstring ('doctests') must be valid Python code,
that in a deterministic way returns the presented output, and that can be
copied and run by users. This can be checked with the script above, and is
also tested on Travis. A failing doctest will be a blocker for merging a PR.

When doing a PR with a docstring update, it is good to post the
output of the validation script in a comment on github.


How to build the Xorbits documentation
--------------------------------------

Requirements
~~~~~~~~~~~~

First, you need to have a development environment to be able to build Xorbits
(see the docs on :ref:`creating a development environment <contributing_environment>`).

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

So how do you build the docs? Navigate to your local
``doc/`` directory in the console and run::

    make html

Then you can find the HTML output in the folder ``doc/build/html/``.

The first time you build the docs, it will take quite a while because it has to run
all the code examples and build all the generated docstring pages. In subsequent
evocations, sphinx will try to only build the pages that have been modified.

If you want to do a full clean build, do::

    make clean
    make html

Open the following file in a web browser to see the full documentation you
just built::

    doc/build/html/index.html

And you'll have the satisfaction of seeing your new and improved documentation!

.. _contributing.dev_docs:

Building main branch documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When pull requests are merged into the Xorbits ``main`` branch, the main parts of
the documentation are also built by readthedocs. These docs are then hosted `here
<https://doc.xorbits.io/en/latest/>`__, see also
the :any:`Continuous Integration <contributing.ci>` section.

Previewing changes
------------------

Once, the pull request is submitted, GitHub Actions will automatically build the
documentation. To view the built site:

#. Wait for the ``docs/readthedocs.org:xorbits`` check to complete.
#. Click ``Details`` for preview.