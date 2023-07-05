<div align="center">
  <img width="77%" alt="" src="https://doc.xorbits.io/en/latest/_static/xorbits.svg"><br>
</div>

[![PyPI Latest Release](https://img.shields.io/pypi/v/xorbits.svg?style=for-the-badge)](https://pypi.org/project/xorbits/)
[![License](https://img.shields.io/pypi/l/xorbits.svg?style=for-the-badge)](https://github.com/xorbitsai/xorbits/blob/main/LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/xorbitsai/xorbits?style=for-the-badge)](https://codecov.io/gh/xorbitsai/xorbits)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xorbitsai/xorbits/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xorbitsai/xorbits/goto?ref=main)
[![Doc](https://readthedocs.org/projects/xorbits/badge/?version=latest&style=for-the-badge)](https://doc.xorbits.io)
[![Slack](https://img.shields.io/badge/join_Slack-781FF5.svg?logo=slack&style=for-the-badge)](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)
[![Twitter](https://img.shields.io/twitter/follow/xorbitsio?logo=twitter&style=for-the-badge)](https://twitter.com/xorbitsio)

## What is Xorbits?

Xorbits is an open-source computing framework that makes it easy to scale data science and machine learning workloads —
from data preprocessing to tuning, training, and model serving. Xorbits can leverage multi-cores or GPUs to accelerate
computation on a single machine or scale out up to thousands of machines to support processing terabytes of data and training or serving large models.

Xorbits provides a suite of best-in-class [libraries](https://doc.xorbits.io/en/latest/libraries/index.html) for data
scientists and machine learning practitioners. Xorbits provides the capability to scale tasks without the necessity for
extensive knowledge of infrastructure.

Xorbits features a familiar Python API that supports a variety of libraries, including pandas, NumPy, PyTorch,
XGBoost, etc. With a simple modification of just one line of code, your pandas workflow can be seamlessly scaled
using Xorbits:

<div align="center">
  <img width="70%" alt="" src="https://raw.githubusercontent.com/xorbitsai/xorbits/main/doc/source/_static/pandas_vs_xorbits.gif"><br>
</div>

## Why Xorbits?

As ML and AI workloads continue to grow in complexity, the computational demands soar high. Even though single-node development
environments like your laptop provide convenience, but they fall short when it comes to accommodating these scaling demands. 

### Seamlessly scale your workflow from laptop to cluster

To use Xorbits, you do not need to specify how to distribute the data or even know how many cores your system has.
You can keep using your existing notebooks and still enjoy a significant speed boost from Xorbits, even on your laptop.

### Process large datasets that pandas can't

Xorbits can [leverage all of your computational cores](https://doc.xorbits.io/en/latest/getting_started/why_xorbits/pandas.html#boosting-performance-and-scalability-with-xorbits). 
It is especially beneficial for handling [larger datasets](https://doc.xorbits.io/en/latest/getting_started/why_xorbits/pandas.html#overcoming-memory-limitations-in-large-datasets-with-xorbits),
where pandas may slow down or run out of memory.

### Lightning-fast speed

According to our benchmark tests, Xorbits surpasses other popular pandas API frameworks in speed and scalability. 
See our [performance comparison](https://doc.xorbits.io/en/latest/getting_started/why_xorbits/comparisons.html#performance-comparison)
and [explanation](https://doc.xorbits.io/en/latest/getting_started/why_xorbits/fast.html).

### Leverage the Python ecosystem with native integrations

Xorbits aims to take full advantage of the entire ML ecosystem, offering native integration with pandas and other libraries.

## Where to get it?
The source code is currently hosted on GitHub at: https://github.com/xorbitsai/xorbits

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/xorbits).

```shell
# PyPI
pip install xorbits
```

## Other resources
* [Documentation](https://doc.xorbits.io)
* [Examples and Tutorials](https://doc.xorbits.io/en/latest/getting_started/examples.html)
* [Performance Benchmarks](https://xorbits.io/benchmark)
* [Development Guide](https://doc.xorbits.io/en/latest/development/index.html)

## License
[Apache 2](LICENSE)

## Roadmaps
The main goals we want to achieve in the future include the following:

* Transitioning from pandas native to arrow native for data storage  
  will reduce the memory cost substantially and is more friendly for compute engine.
* Introducing native engines that leverage technologies like vectorization and codegen 
  to accelerate computations.
* Scale as many libraries and algorithms as possible!

More detailed roadmaps will be revealed soon. Stay tuned!

## Relationship with Mars
The creators of Xorbits are mainly those of Mars, and we currently built Xorbits on Mars 
to reduce duplicated work, but the vision of Xorbits suggests that it's not 
appropriate to put everything on Mars. Instead, we need a new project 
to support the roadmaps better. In the future, we will replace some core internal components 
with other upcoming ones we will propose. Stay tuned!

## Getting involved

| Platform                                                                                      | Purpose                                            |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------|
| [Discourse Forum](https://discuss.xorbits.io)                                                 | Asking usage questions and discussing development. |
| [Github Issues](https://github.com/xorbitsai/xorbits/issues)                                  | Reporting bugs and filing feature requests.        |
| [Slack](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg) | Collaborating with other Xorbits users.            |
| [StackOverflow](https://stackoverflow.com/questions/tagged/xorbits)                           | Asking questions about how to use Xorbits.         |
| [Twitter](https://twitter.com/xorbitsio)                                                      | Staying up-to-date on new features.                |
