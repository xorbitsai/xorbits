<div align="center">
  <img width="77%" alt="" src="https://doc.xorbits.io/en/latest/_static/xorbits.svg"><br>
</div>

# Xorbits: scalable Python data science, familiar & fast.
[![PyPI Latest Release](https://img.shields.io/pypi/v/xorbits.svg?style=for-the-badge)](https://pypi.org/project/xorbits/)
[![License](https://img.shields.io/pypi/l/xorbits.svg?style=for-the-badge)](https://github.com/xprobe-inc/xorbits/blob/main/LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/xprobe-inc/xorbits?style=for-the-badge)](https://codecov.io/gh/xprobe-inc/xorbits)
[![Build Status](https://img.shields.io/github/actions/workflow/status/xprobe-inc/xorbits/python.yaml?branch=main&style=for-the-badge&label=GITHUB%20ACTIONS&logo=github)](https://actions-badge.atrox.dev/xprobe-inc/xorbits/goto?ref=main)
[![Doc](https://readthedocs.org/projects/xorbits/badge/?version=latest&style=for-the-badge)](https://doc.xorbits.io)
[![Slack](https://img.shields.io/badge/join_Slack-781FF5.svg?logo=slack&style=for-the-badge)](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)

## What is it?
Xorbits is a scalable Python data science framework that aims to **scale the whole Python data science world,
including numpy, pandas, scikit-learn and many other libraries**. It can leverage multi cores or GPUs to 
accelerate computation on a single machine, or scale out up to thousands of machines to support processing 
terabytes of data. In our benchmark test, **Xorbits is the fastest framework among 
the most popular distributed data science frameworks**.

As for the name of `xorbits`, it has many meanings, you can treat it as `X-or-bits` or `X-orbits` or `xor-bits`, 
just have fun to comprehend it in your own way.

## At a glance

<div align="center">
  <img width="100%" alt="" src="https://raw.githubusercontent.com/xprobe-inc/xorbits/main/doc/source/_static/pandas_vs_xorbits.gif"><br>
</div>

Codes are almost identical except for the import, 
replace `import pandas` with `import xorbits.pandas` will just work, 
so does numpy and so forth.

## Where to get it
The source code is currently hosted on GitHub at: https://github.com/xprobe-inc/xorbits

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/xorbits).

```shell
# PyPI
pip install xorbits
```

## API compatibility
As long as you know how to use numpy, pandas and so forth, you would probably know how to use Xorbits.

All Xorbits APIs implemented or planned include:

| API                                                                            | Implemented version or plan |
|--------------------------------------------------------------------------------|-----------------------------|
| [xorbits.pandas](https://doc.xorbits.io/en/latest/reference/pandas/index.html) | v0.1.0                      |
| [xorbits.numpy](https://doc.xorbits.io/en/latest/reference/numpy/index.html)   | v0.1.0                      |
| xorbits.sklearn                                                                | Planned in the near future  |
| xorbits.xgboost                                                                | Planned in the near future  |
| xorbits.lightgbm                                                               | Planned in the near future  |
| xorbits.xarray                                                                 | Planned in the future       |

## Lightning fast speed
Xorbits is the fastest compared to other popular frameworks according to our benchmark tests.

We did benchmarks for TPC-H at scale factor 100 (~100 GB datasets) and 1000 (~1 TB datasets). 
The performances are shown as below.

### TPC-H SF100: Xorbits vs Dask

![Xorbits vs Dask](https://xorbits.io/res/benchmark_dask.png)

Q21 was excluded since Dask ran out of memory. Across all queries, Xorbits was found to be 7.3x faster than Dask.

### TPC-H SF100: Xorbits vs Pandas API on Spark

![Xorbits vs Pandas API on Spark](https://xorbits.io/res/benchmark_spark.png)

Across all queries, the two systems have roughly similar performance, but Xorbits provided much better API compatibility.
Pandas API on Spark failed on Q1, Q4, Q7, Q21, and ran out of memory on Q20.

### TPC-H SF100: Xorbits vs Modin

![Xorbits vs Modin](https://xorbits.io/res/benchmark_modin.png)

Although Modin ran out of memory for most of the queries that involve heavy data shuffles, 
making the performance difference less obvious, Xorbits was still found to be 3.2x faster than Modin.

### TPC-H SF1000: Xorbits

![Xorbits](https://xorbits.io/res/xorbits_1t.png)

Although Xorbits is able to pass all the queries in a row, 
Dask, Pandas API on Spark and Modin failed on most of the queries. 
Thus, we are not able to compare the performance difference now, and we plan to try again later.

For more information, see [performance benchmarks](https://xorbits.io/benchmark).

## Deployment

Xorbits can be deployed on your local machine, largely deployed to a cluster via command lines,
or deploy via Kubernetes and clouds.


| Deployment                                                                | Description                                                |
|---------------------------------------------------------------------------|------------------------------------------------------------|
| [Local](https://doc.xorbits.io/en/latest/deployment/local.html)           | Running Xorbits on a local machine, e.g. laptop            |
| [Cluster](https://doc.xorbits.io/en/latest/deployment/cluster.html)       | Deploy Xorbits to existing cluster via command lines       |
| [Kubernetes](https://doc.xorbits.io/en/latest/deployment/kubernetes.html) | Deploy Xorbits to existing k8s cluster via python codes    |
| [Cloud](https://doc.xorbits.io/en/latest/deployment/cloud.html)           | Deploy Xorbits to various cloud platforms via python codes |


## License
[Apache 2](LICENSE)

## Documentation
The official documentation is hosted on: https://doc.xorbits.io

## Roadmaps
Main goals we want to achieve in the future include:

* Transitioning from pandas native to arrow native for data storage,  
  it will reduce the memory cost substantially and is more friendly for compute engine.
* Introducing native engines that leverage technologies like vectorization and codegen 
  to accelerate computations.
* Scale as many libraries and algorithms as possible!

More detailed roadmaps will be revealed soon, stay tuned!

## Relationship with Mars
The creators of Xorbits are mainly those of Mars, we built Xorbits currently on Mars 
to reduce duplicated work, but the vision of Xorbits suggests that it's not 
appropriate to put everything into Mars, instead, we need a new project 
to support the roadmaps better. In the future, we will replace some core internal components 
with other upcoming ones we will propose, stay tuned!

## Getting involved

| Platform                                                                                       | Purpose                                            |
|------------------------------------------------------------------------------------------------|----------------------------------------------------|
| [Discourse Forum](https://discuss.xorbits.io)                                                  | Asking usage questions and discussing development. |
| [Github Issues](https://github.com/xprobe-inc/xorbits/issues)                                  | Reporting bugs and filing feature requests.        |
| [Slack](https://join.slack.com/t/xorbitsio/shared_invite/zt-1o3z9ucdh-RbfhbPVpx7prOVdM1CAuxg)  | Collaborating with other Xorbits users.            |
| [StackOverflow](https://stackoverflow.com/questions/tagged/xorbits)                            | Asking questions about how to use Xorbits.         |
| [Twitter](https://twitter.com/xorbitsio)                                                       | Staying up-to-date on new features.                |
