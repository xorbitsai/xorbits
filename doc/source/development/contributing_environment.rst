.. _contributing_environment:

==================================
Creating a development environment
==================================

To test out code changes, you'll need to build Xorbits from source, which
requires a C/C++ compiler and Python environment. If you're making documentation
changes, you can skip to :ref:`contributing to the documentation <contributing_documentation>` but if you skip
creating the development environment you won't be able to build the documentation
locally before pushing your changes. It's recommended to also install the :ref:`pre-commit hooks <contributing.pre-commit>`.

.. contents:: Table of contents:
   :local:

Step 1: install a C compiler
----------------------------

How to do this will depend on your platform. If you choose to user ``Docker``
in the next step, then you can skip this step.

**Windows**

You will need `Build Tools for Visual Studio 2022
<https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022>`_.

.. note::
        You DO NOT need to install Visual Studio 2022.
        You only need "Build Tools for Visual Studio 2022" found by
        scrolling down to "All downloads" -> "Tools for Visual Studio".
        In the installer, select the "Desktop development with C++" Workloads.

Alternatively, you can install the necessary components on the commandline using
`vs_BuildTools.exe <https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?source=recommendations&view=vs-2022>`_

Alternatively, you could use the `WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_
and consult the ``Linux`` instructions below.

**macOS**

To use the :ref:`mamba <contributing.mamba>`-based compilers, you will need to install the
Developer Tools using ``xcode-select --install``. Otherwise
information about compiler installation can be found here:
https://devguide.python.org/setup/#macos

**Linux**

For Linux-based :ref:`mamba <contributing.mamba>` installations, you won't have to install any
additional components outside of the mamba environment. The instructions
below are only needed if your setup isn't based on mamba environments.

Some Linux distributions will come with a pre-installed C compiler. To find out
which compilers (and versions) are installed on your system::

    # for Debian/Ubuntu:
    dpkg --list | grep compiler
    # for Red Hat/RHEL/CentOS/Fedora:
    yum list installed | grep -i --color compiler

`GCC (GNU Compiler Collection) <https://gcc.gnu.org/>`_, is a widely used
compiler, which supports C and a number of other languages. If GCC is listed
as an installed compiler nothing more is required.

If no C compiler is installed, or you wish to upgrade, or you're using a different
Linux distribution, consult your favorite search engine for compiler installation/update
instructions.

Let us know if you have any difficulties by opening an issue or reaching out on our contributor
community, join slack in `Community <https://xorbits.io/community>`_.

Step 2: create an isolated environment
----------------------------------------

Before we begin, please:

* Make sure that you have :any:`cloned the repository <contributing.forking>`
* ``cd`` to the xorbits source directory

.. _contributing.mamba:

Option 1: using mamba (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Install `mamba <https://mamba.readthedocs.io/en/latest/installation.html>`_
* Make sure your mamba is up to date (``mamba update mamba``)

.. code-block:: none

   # Create and activate the build environment
   mamba env create --file environment.yml
   mamba activate xorbits

Option 2: using pip
~~~~~~~~~~~~~~~~~~~

You'll need to have at least the :ref:`minimum Python version <install.version>` that Xorbits supports.
You also need to have ``setuptools`` 51.0.0 or later to build Xorbits.

**Unix**/**macOS with virtualenv**

.. code-block:: bash

   # Create a virtual environment
   # Use an ENV_DIR of your choice. We'll use ~/virtualenvs/xorbits-dev
   # Any parent directories should already exist
   python3 -m venv ~/virtualenvs/xorbits-dev

   # Activate the virtualenv
   . ~/virtualenvs/xorbits-dev/bin/activate

   # Install the build dependencies
   python -m pip install -r requirements-dev.txt

**Unix**/**macOS with pyenv**

Consult the docs for setting up pyenv `here <https://github.com/pyenv/pyenv>`__.

.. code-block:: bash

   # Create a virtual environment
   # Use an ENV_DIR of your choice. We'll use ~/Users/<yourname>/.pyenv/versions/xorbits-dev
   pyenv virtualenv <version> <name-to-give-it>

   # For instance:
   pyenv virtualenv 3.9.10 xorbits-dev

   # Activate the virtualenv
   pyenv activate xorbits-dev

   # Now install the build dependencies in the cloned Xorbits repo
   python -m pip install -r requirements-dev.txt

**Windows**

Below is a brief overview on how to set-up a virtual environment with Powershell
under Windows. For details please refer to the
`official virtualenv user guide <https://virtualenv.pypa.io/en/latest/user_guide.html#activators>`__.

Use an ENV_DIR of your choice. We'll use ``~\\virtualenvs\\xorbits-dev`` where
``~`` is the folder pointed to by either ``$env:USERPROFILE`` (Powershell) or
``%USERPROFILE%`` (cmd.exe) environment variable. Any parent directories
should already exist.

.. code-block:: powershell

   # Create a virtual environment
   python -m venv $env:USERPROFILE\virtualenvs\xorbits-dev

   # Activate the virtualenv. Use activate.bat for cmd.exe
   ~\virtualenvs\xorbits-dev\Scripts\Activate.ps1

   # Install the build dependencies
   python -m pip install -e ".[dev]"

Option 3: using Docker
~~~~~~~~~~~~~~~~~~~~~~

Xorbits provides a ``DockerFile`` in the root directory to build a Docker image
with a full Xorbits development environment.

**Docker Commands**

Build the Docker image::

    # Build the image
    docker build -t xorbits-dev .

Run Container::

    # Run a container and bind your local repo to the container
    # This command assumes you are running from your local repo
    # but if not alter ${PWD} to match your local repo path
    docker run -it --rm xorbits-dev /bin/bash

*Even easier, you can integrate Docker with the following IDEs:*

**Visual Studio Code**

You can use the DockerFile to launch a remote session with Visual Studio Code,
a popular free IDE, using the ``.devcontainer.json`` file.
See https://code.visualstudio.com/docs/remote/containers for details.

**PyCharm (Professional)**

Enable Docker support and use the Services tool window to build and manage images as well as
run and interact with containers.
See https://www.jetbrains.com/help/pycharm/docker.html for details.

Step 3: build and install Xorbits
---------------------------------

You can now run::

   # Build and install Xorbits
   python setup.py build_ext -j 4
   python -m pip install -e . --no-build-isolation --no-use-pep517

At this point you should be able to import Xorbits from your locally built version::

   $ python
   >>> import xorbits
   >>> print(xorbits.__version__)  # note: the exact output may differ
   0.1.1+20.g9b58334.dirty

This will create the new environment, and not touch any of your existing environments,
nor any existing Python installation.

.. note::
   You will need to repeat this step each time the C extensions change,
   or if you did a fetch and merge from ``upstream/main``.