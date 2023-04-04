.. _logging:

=======
Logging
=======

This document will explain Xorbits’s logging system. By default, Xorbits logs are written to stderr
as well as files.

Log to stderr
-------------
The default log level for stderr is set to ``WARNING`` to limit the amount of log output.

In a :ref:`local deployment <deployment_local>`, you can modify the log level and format before
initializing Xorbits as follows:

.. code-block:: python

    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    import xorbits
    # Logs ...
    xorbits.init()
    # Logs ...

In a :ref:`cluster deployment <deployment_cluster>`, you can modify the log level and format by
specifying a log configuration when starting Xorbits supervisor and workers::

    python -m xorbits.supervisor \
           -H <supervisor_ip> \
           -p <supervisor_port> \
           -w <web_port> \
           --log-config /path/to/logging.conf

To create your own log configuration, please refer to the log configuration
:ref:`template <log_config_template>`.

Log to files
------------
By default, Xorbits’ logs are saved in the directory ``/tmp/xorbits/logs`` for Linux and Mac OS,
and ``C:\Temp\xorbits\logs`` for Windows. The default log level for files is set to ``DEBUG`` for
troubleshooting purposes.

.. note::
    Since ``v0.3.0``, Xorbits’ logs are saved in the directory ``/tmp/${USER}/xorbits/logs`` for Linux and Mac OS,
    and ``C:\Temp\${USERNAME}\xorbits\logs`` for Windows by default.

Local deployment
~~~~~~~~~~~~~~~~
In a :ref:`local deployment <deployment_local>`, Xorbits creates a subdirectory under the log
directory. The name of the subdirectory corresponds to the process startup time in nanoseconds. In
a local deployment, the logs of Xorbits supervisor and Xorbits workers are combined into a single
file. An example of the log directory structure is shown below::

    /tmp/xorbits
    └── logs
        └── 1679904140623326000
            └── xorbits.log

.. note::
    Since ``v0.3.0``, an example of the log directory structure for user ``test`` is shown below::

        /tmp/test/xorbits
        └── logs
            └── 1679904140623326000
                └── xorbits.log

If you wish to change the log directory, level, or format, specify them when calling
``xorbits.init()``:

.. code-block:: python

    import xorbits
    xorbits.init(
        log_config={
            "log_dir": "/path/to/logs",
            "level": "WARNING",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    )

Cluster deployment
~~~~~~~~~~~~~~~~~~
In a :ref:`cluster deployment <deployment_cluster>`, Xorbits supervisor and Xorbits workers each
create their own subdirectory under the log directory. The name of the subdirectory starts with the
role name, followed by the process startup time in nanoseconds. An example of the log directory
structure is shown below::

    /tmp/xorbits
    └── logs
        ├── supervisor_1679923647642312000
        │   └── xorbits.log
        └── worker_1679923657597859000
            └── xorbits.log

.. note::
    Since ``v0.3.0``, an example of the log directory structure for user ``test`` is shown below::

        /tmp/test/xorbits
        └── logs
            ├── supervisor_1679923647642312000
            │   └── xorbits.log
            └── worker_1679923657597859000
                └── xorbits.log

You can easily modify the log level, format, or directory with command line arguments. For
instance::

    python -m xorbits.supervisor \
           -H <supervisor_ip> \
           -p <supervisor_port> \
           -w <web_port> \
           --log-level INFO
           --log-format '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
           --log-dir /path/to/logs

Log rotation
------------
Xorbits supports log rotation of log files. By default, logs rotate when they reach 100MB
(maxBytes), and up to 30 backup files (backupCount) are kept.

.. _log_config_template:

Log configuration template
--------------------------
Here's a log configuration template::

    [loggers]
    keys=root,main,deploy,services,oscar,tornado,dataframe,learn,tensor,xorbits_core,xorbits_deploy,xorbits_numpy,xorbits_pandas,xorbits_remote,xorbits_web

    [handlers]
    keys=stream_handler,file_handler

    [formatters]
    keys=formatter

    [logger_root]
    level=WARN
    handlers=stream_handler,file_handler

    [logger_main]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=__main__
    propagate=0

    [logger_deploy]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits._mars.deploy
    propagate=0

    [logger_oscar]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits._mars.oscar
    propagate=0

    [logger_services]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits._mars.services
    propagate=0

    [logger_dataframe]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits._mars.dataframe
    propagate=0

    [logger_learn]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits._mars.learn
    propagate=0

    [logger_tensor]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits._mars.tensor
    propagate=0

    [logger_tornado]
    level=WARN
    handlers=stream_handler,file_handler
    qualname=tornado
    propagate=0

    [logger_xorbits_core]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits.core
    propagate=0

    [logger_xorbits_deploy]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits.deploy
    propagate=0

    [logger_xorbits_numpy]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits.numpy
    propagate=0

    [logger_xorbits_pandas]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits.pandas
    propagate=0

    [logger_xorbits_remote]
    level=DEBUG
    handlers=stream_handler,file_handler
    qualname=xorbits.remote
    propagate=0

    [logger_xorbits_web]
    level=WARN
    handlers=stream_handler,file_handler
    qualname=xorbits.web
    propagate=0

    [handler_stream_handler]
    class=StreamHandler
    formatter=formatter
    level=WARN
    args=(sys.stderr,)

    [handler_file_handler]
    class=logging.handlers.RotatingFileHandler
    formatter=formatter
    level=DEBUG
    args=('/path/to/logs/xorbits.log',)
    kwargs={'mode': 'a', 'maxBytes': 104857600, 'backupCount': 30}

    [formatter_formatter]
    format=%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s
