.. _contributing_environment:

==================================
Creating a development environment
==================================

To test out code changes, you'll need to build Xorbits from source, which
requires a C/C++ compiler, Node.js, and Python environment. It's recommended to also install
the :ref:`pre-commit hooks <contributing.pre-commit>`.

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

To use the :ref:`conda <contributing.conda>`-based compilers, you will need to install the
Developer Tools using ``xcode-select --install``. Otherwise
information about compiler installation can be found here:
https://devguide.python.org/setup/#macos

**Linux**

For Linux-based :ref:`conda <contributing.conda>` installations, you won't have to install any
additional components outside of the conda environment. The instructions
below are only needed if your setup isn't based on conda environments.

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

Step 2: install Node.js
-----------------------

To build Xorbits web UI, you will need `Node.js <https://nodejs.org/en>`_. It is recommended to
install Node.js with `nvm <https://github.com/nvm-sh/nvm>`_ .

.. note::
        The minimum supported Node.js version is 18.

Step 3: create an isolated environment
--------------------------------------

Before we begin, please:

* Make sure that you have :any:`cloned the repository <contributing.forking>`
* ``cd`` to the xorbits source directory

.. _contributing.conda:

Option 1: using conda (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Install `conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_

.. code-block:: none

   # Create and activate the build environment
   conda create --name xorbits-dev python=3.10
   conda activate xorbits-dev

Option 2: using Docker
~~~~~~~~~~~~~~~~~~~~~~

Unless you have a specific requirement to install additional Python libraries,
it is **highly recommended** to use the Xorbits image available
on our `Dockerhub <https://hub.docker.com/repository/docker/xprobe/xorbits/general>`_.
It includes the complete environment required to run Xorbits.

The images available on Dockerhub include versions for all supported Python versions, with the suffix ``py<python_version>``.
For the image tag prefixes, ``nightly-main`` represents the latest code from `Xorbits GitHub repository <https://github.com/xorbitsai/xorbits>`_ on a daily basis,
while ``v<release_version>`` represents version numbers for each release.
You can choose to pull the image based on your specific requirements.

If you indeed need to manually build Xorbits image, Xorbits provides a ``DockerFile`` in the ``python/xorbits/deploy/docker`` directory to build a Docker image
with a full Xorbits development environment.

**Docker Commands**

Build the Docker image::

    # Switch the current working directory to the top-level "xorbits" directory
    $ cd xorbits

    # Build the image
    docker build -t xorbits-dev --progress=plain -f python/xorbits/deploy/docker/Dockerfile . --build-arg PYTHON_VERSION=<your_python_version>

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

Step 4: build and install Xorbits
---------------------------------

You can now run::

   # Build and install Xorbits
   python -m pip install -e ".[dev]"
   python setup.py build_ext -i
   python setup.py build_web

At this point you should be able to import Xorbits from your locally built version::

   $ python
   >>> import xorbits
   >>> print(xorbits.__version__)  # note: the exact output may differ
   0.1.1+20.g9b58334.dirty

This will create the new environment, and not touch any of your existing environments,
nor any existing Python installation.

.. note::
   You will need to repeat this step each time the web UI or C extensions change,
   or if you did a fetch and merge from ``upstream/main``.