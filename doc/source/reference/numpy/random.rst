Random Sampling
===============

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

Sample random data
------------------

+--------------------------+--------------------+------------------------+----------------------------------+
| ``xorbits.numpy.random`` | ``numpy.random``   | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------------+--------------------+------------------------+----------------------------------+
| ``rand``                 | `rand`_            | Y                      |                                  |
+--------------------------+--------------------+------------------------+----------------------------------+
| ``randn``                | `randn`_           | Y                      |                                  |
+--------------------------+--------------------+------------------------+----------------------------------+
| ``randint``              | `randint`_         | Y                      |                                  |
+--------------------------+--------------------+------------------------+----------------------------------+
| ``random_integers``      | `random_integers`_ | Y                      |                                  |
+--------------------------+--------------------+------------------------+----------------------------------+
| ``random_sample``        | `random_sample`_   | Y                      |                                  |
+--------------------------+--------------------+------------------------+----------------------------------+
| ``random``               | `random`_          | Y                      |                                  |
+--------------------------+--------------------+------------------------+----------------------------------+
| ``ranf``                 | `ranf`_            | Y                      |                                  |
+--------------------------+--------------------+------------------------+----------------------------------+
| ``sample``               | `sample`_          | Y                      |                                  |
+--------------------------+--------------------+------------------------+----------------------------------+
| ``choice``               | `choice`_          | Y                      |                                  |
+--------------------------+--------------------+------------------------+----------------------------------+
| ``bytes``                | `bytes`_           | Y                      |                                  |
+--------------------------+--------------------+------------------------+----------------------------------+

Distributions
-------------

+--------------------------+-------------------------+------------------------+----------------------------------+
| ``xorbits.numpy.random`` | ``numpy.random``        | Implemented? (Y/N/P/D) | Notes for Current implementation |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``beta``                 | `beta`_                 | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``binomial``             | `binomial`_             | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``chisquare``            | `chisquare`_            | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``dirichlet``            | `dirichlet`_            | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``exponential``          | `exponential`_          | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``f``                    | `f`_                    | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``gamma``                | `gamma`_                | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``geometric``            | `geometric`_            | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``gumbel``               | `gumbel`_               | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``hypergeometric``       | `hypergeometric`_       | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``laplace``              | `laplace`_              | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``lognormal``            | `lognormal`_            | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``logseries``            | `logseries`_            | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``multinomial``          | `multinomial`_          | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``multivariate_normal``  | `multivariate_normal`_  | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``negative_binomial``    | `negative_binomial`_    | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``noncentral_chisquare`` | `noncentral_chisquare`_ | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``noncentral_f``         | `noncentral_f`_         | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``normal``               | `normal`_               | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``pareto``               | `pareto`_               | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``poisson``              | `poisson`_              | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``power``                | `power`_                | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``rayleigh``             | `rayleigh`_             | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``standard_cauchy``      | `standard_cauchy`_      | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``standard_exponential`` | `standard_exponential`_ | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``standard_gamma``       | `standard_gamma`_       | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``standard_normal``      | `standard_normal`_      | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``standard_t``           | `standard_t`_           | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``triangular``           | `triangular`_           | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``uniform``              | `uniform`_              | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``vonmises``             | `vonmises`_             | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``wald``                 | `wald`_                 | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``weibull``              | `weibull`_              | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+
| ``zipf``                 | `zipf`_                 | Y                      |                                  |
+--------------------------+-------------------------+------------------------+----------------------------------+

Random number generator
-----------------------

+--------------------------+------------------+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``xorbits.numpy.random`` | ``numpy.random`` | Implemented? (Y/N/P/D) | Notes for Current implementation                                                                                                                                                                   |
+--------------------------+------------------+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``seed``                 | `seed`_          | Y                      |                                                                                                                                                                                                    |
+--------------------------+------------------+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``RandomState``          | `RandomState`_   | Y                      | ``RandomState`` ensures backward compatibility with NumPy 1.16, reproducing identical sequences. It is frozen with no further updates and should only be used for consistency with older versions. |
+--------------------------+------------------+------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`rand`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
.. _`randn`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
.. _`randint`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
.. _`random_integers`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_integers.html
.. _`random_sample`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_sample.html
.. _`random`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html
.. _`ranf`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.ranf.html
.. _`sample`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.sample.html
.. _`choice`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html
.. _`bytes`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.bytes.html
.. _`beta`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html
.. _`binomial`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
.. _`chisquare`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.chisquare.html
.. _`dirichlet`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.dirichlet.html
.. _`exponential`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html
.. _`f`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.f.html
.. _`gamma`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html
.. _`geometric`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html
.. _`gumbel`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html
.. _`hypergeometric`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.hypergeometric.html
.. _`laplace`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.laplace.html
.. _`lognormal`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html
.. _`logseries`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.logseries.html
.. _`multinomial`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html
.. _`multivariate_normal`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html
.. _`negative_binomial`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.negative_binomial.html
.. _`noncentral_chisquare`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_chisquare.html
.. _`noncentral_f`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_f.html
.. _`normal`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
.. _`pareto`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.pareto.html
.. _`poisson`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html
.. _`power`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.power.html
.. _`rayleigh`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rayleigh.html
.. _`standard_cauchy`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_cauchy.html
.. _`standard_exponential`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_exponential.html
.. _`standard_gamma`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_gamma.html
.. _`standard_normal`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html
.. _`standard_t`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_t.html
.. _`triangular`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html
.. _`uniform`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html
.. _`vonmises`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.vonmises.html
.. _`wald`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.wald.html
.. _`weibull`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.weibull.html
.. _`zipf`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.zipf.html
.. _`seed`: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
.. _`RandomState`: https://numpy.org/doc/stable/reference/random/legacy.html#numpy.random.RandomState
