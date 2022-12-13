<div align="center">
  <img src="https://raw.githubusercontent.com/xprobe-inc/xorbits/main/doc/source/_static/xorbits.png"><br>
</div>

# Xorbits: scalable Python data science, familiar & fast.
[![PyPI Latest Release](https://img.shields.io/pypi/v/xorbits.svg?style=for-the-badge)](https://pypi.org/project/xorbits/)
[![License](https://img.shields.io/pypi/l/xorbits.svg?style=for-the-badge)](https://github.com/xprobe-inc/xorbits/blob/main/LICENSE)
[![Coverage](https://img.shields.io/codecov/c/github/xprobe-inc/xorbits?style=for-the-badge)](https://codecov.io/gh/xprobe-inc/xorbits)
[![Slack](https://img.shields.io/badge/join_Slack-information-brightgreen.svg?logo=slack&style=for-the-badge)](https://slack.xorbits.io)

## What is it?
Xorbits is a scalable Python data science framework that aims to **scale the whole Python data science world,
including numpy, pandas, scikit-learn and many other libraries**. It can leverage the multi-cores or GPUs to 
accelerate computation on a single machine, or scale to up to thousands of machines to support processing 
terabytes of data. In our benchmark test, **Xorbits is the fastest framework among 
the most popular distributed data science frameworks**.

As for the name of `xorbits`, it has many meanings, you can treat it as `X-or-bits` or `X-orbits` or `xor-bits`, 
just have fun to comprehend it in your own way.

## Where to get it
The source code is currently hosted on GitHub at: https://github.com/xprobe-inc/xorbits

Binary installers for the latest released version are available at the [Python
Package Index (PyPI)](https://pypi.org/project/xorbits).

```shell
# PyPI
pip install xorbits
```

Deploy to bare metal machines is dead easy as well. Ensure xorbits is installed on each machine,
then

For supervisor which schedules tasks to workers.

```shell
# for supervisor
python -m xorbits.supervisor -H <host_name> -p <supervisor_port> -w <web_port>
```

For workers which run actual computatins.

```shell
# for worker
python -m xorbits.worker -H <host_name> -p <worker_port> -s <supervisor_ip>:<supervisor_port>
```

## API compatibility
As long as you know how to use numpy, pandas and so forth, you would probably know how to use xorbits.

Here is an example.

<table>
<tr>
<td> pandas </td> <td> Xorbits </td>
</tr>
<tr>
<td>

```python
import pandas as pd

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

m = ratings.groupby(
    'MOVIE_ID', as_index=False).agg(
    {'RATING': ['mean', 'count']})
m.columns = ['MOVIE_ID', 'RATING', 'COUNT']
m = m[m['COUNT'] > 100]
top_100 = m.sort_values(
    'RATING', ascending=False)[:100]
top_100 = top_100.merge(
    movies[['MOVIE_ID', 'NAME']])
print(top_100)
```

</td>
<td>
    
```python
import xorbits.pandas as pd

ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

m = ratings.groupby(
    'MOVIE_ID', as_index=False).agg(
    {'RATING': ['mean', 'count']})
m.columns = ['MOVIE_ID', 'RATING', 'COUNT']
m = m[m['COUNT'] > 100]
top_100 = m.sort_values(
    'RATING', ascending=False)[:100]
top_100 = top_100.merge(
    movies[['MOVIE_ID', 'NAME']])
print(top_100)
```
</td>
</tr>
</table>

Codes are almost identical except for the import, 
replace `import pandas` with `import xorbits.pandas` will just work, 
so does numpy and so forth.

## Lightning fast speed
Xorbits is fastest compared to other popular frameworks.

We did a benchmark for TPC-H at scale factor 100 and 1000. 
The performance showed as below.

## License
[Apache 2](LICENSE)

## Documentation
The official documentation is hosted on: https://doc.xorbits.io

## Getting involved

<table>
<tr>
<td> Platform </td> <td> Purpose </td>
</tr>
<tr>
<td>
<a href="https://discuss.xorbits.io">Discourse Forum</a>
</td>
<td>
Asking usage questions and discussing development.
</td>
</tr>
<tr>
<td>
<a href="https://github.com/xprobe-inc/xorbits/issues">Github Issues</a>
</td>
<td>
Reporting bugs and filing feature requests.
</td>
</tr>
<tr>
<td>
<a href="https://slack.xorbits.io">Slack</a>
</td>
<td>
Collaborating with other Xorbits users.
</td>
</tr>
<tr>
<td>
<a href="https://stackoverflow.com/questions/tagged/xorbits">StackOverflow</a>
</td>
<td>
Asking questions about how to use Xorbits.
</td>
</tr>
<tr>
<td>
<a href="https://twitter.com/xorbits">Twitter</a>
</td>
<td>
Staying up-to-date on new features.
</td>
</tr>
</table>
