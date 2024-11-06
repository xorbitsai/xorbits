Mathematical functions
======================

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

Trigonometric functions
-----------------------

+-------------------+------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``  | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+------------+------------------------+----------------------------------+
| ``sin``           | `sin`_     | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``cos``           | `cos`_     | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``tan``           | `tan`_     | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``arcsin``        | `arcsin`_  | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``arccos``        | `arccos`_  | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``arctan``        | `arctan`_  | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``hypot``         | `hypot`_   | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``arctan2``       | `arctan2`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``degrees``       | `degrees`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``radians``       | `radians`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``unwrap``        | `unwrap`_  | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``deg2rad``       | `deg2rad`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``rad2deg``       | `rad2deg`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+

Hyperbolic functions
--------------------

+-------------------+------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``  | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+------------+------------------------+----------------------------------+
| ``sinh``          | `sinh`_    | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``cosh``          | `cosh`_    | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``tanh``          | `tanh`_    | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``arcsinh``       | `arcsinh`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``arccosh``       | `arccosh`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``arctanh``       | `arctanh`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+

Rounding
--------

+-------------------+-----------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------+------------------------+----------------------------------+
| ``around``        | `around`_ | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``rint``          | `rint`_   | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``fix``           | `fix`_    | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``floor``         | `floor`_  | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``ceil``          | `ceil`_   | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``trunc``         | `trunc`_  | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+

Sums, products, differences
---------------------------

+-------------------+---------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``     | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+---------------+------------------------+----------------------------------+
| ``prod``          | `prod`_       | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``sum``           | `sum`_        | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``nanprod``       | `nanprod`_    | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``nansum``        | `nansum`_     | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``cumprod``       | `cumprod`_    | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``cumsum``        | `cumsum`_     | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``nancumprod``    | `nancumprod`_ | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``nancumsum``     | `nancumsum`_  | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``diff``          | `diff`_       | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``ediff1d``       | `ediff1d`_    | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``gradient``      | `gradient`_   | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``cross``         | `cross`_      | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``trapz``         | `trapz`_      | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+

Exponents and logarithms
------------------------

+-------------------+---------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``     | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+---------------+------------------------+----------------------------------+
| ``exp``           | `exp`_        | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``expm1``         | `expm1`_      | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``exp2``          | `exp2`_       | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``log``           | `log`_        | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``log10``         | `log10`_      | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``log2``          | `log2`_       | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``log1p``         | `log1p`_      | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``logaddexp``     | `logaddexp`_  | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+
| ``logaddexp2``    | `logaddexp2`_ | Y                      |                                  |
+-------------------+---------------+------------------------+----------------------------------+

Other special functions
-----------------------

+-------------------+-----------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------+------------------------+----------------------------------+
| ``i0``            | `i0`_     | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``sinc``          | `sinc`_   | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+

Floating point routines
-----------------------

+-------------------+--------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``    | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+--------------+------------------------+----------------------------------+
| ``signbit``       | `signbit`_   | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``copysign``      | `copysign`_  | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``frexp``         | `frexp`_     | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``ldexp``         | `ldexp`_     | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``nextafter``     | `nextafter`_ | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``spacing``       | `spacing`_   | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+

Rational routines
-----------------

+-------------------+-----------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------+------------------------+----------------------------------+
| ``lcm``           | `lcm`_    | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+
| ``gcd``           | `gcd`_    | Y                      |                                  |
+-------------------+-----------+------------------------+----------------------------------+

Arithmetic operations
---------------------

+-------------------+-----------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``       | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+-----------------+------------------------+----------------------------------+
| ``add``           | `add`_          | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``reciprocal``    | `reciprocal`_   | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``positive``      | `positive`_     | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``negative``      | `negative`_     | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``multiply``      | `multiply`_     | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``divide``        | `divide`_       | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``power``         | `power`_        | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``subtract``      | `subtract`_     | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``true_divide``   | `true_divide`_  | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``floor_divide``  | `floor_divide`_ | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``float_power``   | `float_power`_  | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``fmod``          | `fmod`_         | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``mod``           | `mod`_          | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``modf``          | `modf`_         | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``remainder``     | `remainder`_    | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+
| ``divmod``        | `divmod`_       | Y                      |                                  |
+-------------------+-----------------+------------------------+----------------------------------+

Handling complex numbers
------------------------

+-------------------+--------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``    | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+--------------+------------------------+----------------------------------+
| ``angle``         | `angle`_     | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``real``          | `real`_      | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``imag``          | `imag`_      | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``conj``          | `conj`_      | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+
| ``conjugate``     | `conjugate`_ | Y                      |                                  |
+-------------------+--------------+------------------------+----------------------------------+

Extrema Finding
---------------

+-------------------+------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``  | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+------------+------------------------+----------------------------------+
| ``maximum``       | `maximum`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``fmax``          | `fmax`_    | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``amax``          | `amax`_    | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``nanmax``        | `nanmax`_  | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``minimum``       | `minimum`_ | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``fmin``          | `fmin`_    | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``amin``          | `amin`_    | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+
| ``nanmin``        | `nanmin`_  | Y                      |                                  |
+-------------------+------------+------------------------+----------------------------------+

Miscellaneous
-------------

+-------------------+------------------+------------------------+----------------------------------+
| ``xorbits.numpy`` | ``numpy``        | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-------------------+------------------+------------------------+----------------------------------+
| ``convolve``      | `convolve`_      | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``clip``          | `clip`_          | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``sqrt``          | `sqrt`_          | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``cbrt``          | `cbrt`_          | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``square``        | `square`_        | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``absolute``      | `absolute`_      | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``fabs``          | `fabs`_          | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``sign``          | `sign`_          | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``heaviside``     | `heaviside`_     | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``nan_to_num``    | `nan_to_num`_    | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``real_if_close`` | `real_if_close`_ | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+
| ``interp``        | `interp`_        | Y                      |                                  |
+-------------------+------------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`sin`: https://numpy.org/doc/stable/reference/generated/numpy.sin.html
.. _`cos`: https://numpy.org/doc/stable/reference/generated/numpy.cos.html
.. _`tan`: https://numpy.org/doc/stable/reference/generated/numpy.tan.html
.. _`arcsin`: https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html
.. _`arccos`: https://numpy.org/doc/stable/reference/generated/numpy.arccos.html
.. _`arctan`: https://numpy.org/doc/stable/reference/generated/numpy.arctan.html
.. _`hypot`: https://numpy.org/doc/stable/reference/generated/numpy.hypot.html
.. _`arctan2`: https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
.. _`degrees`: https://numpy.org/doc/stable/reference/generated/numpy.degrees.html
.. _`radians`: https://numpy.org/doc/stable/reference/generated/numpy.radians.html
.. _`unwrap`: https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html
.. _`deg2rad`: https://numpy.org/doc/stable/reference/generated/numpy.deg2rad.html
.. _`rad2deg`: https://numpy.org/doc/stable/reference/generated/numpy.rad2deg.html
.. _`sinh`: https://numpy.org/doc/stable/reference/generated/numpy.sinh.html
.. _`cosh`: https://numpy.org/doc/stable/reference/generated/numpy.cosh.html
.. _`tanh`: https://numpy.org/doc/stable/reference/generated/numpy.tanh.html
.. _`arcsinh`: https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html
.. _`arccosh`: https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html
.. _`arctanh`: https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html
.. _`around`: https://numpy.org/doc/stable/reference/generated/numpy.around.html
.. _`rint`: https://numpy.org/doc/stable/reference/generated/numpy.rint.html
.. _`fix`: https://numpy.org/doc/stable/reference/generated/numpy.fix.html
.. _`floor`: https://numpy.org/doc/stable/reference/generated/numpy.floor.html
.. _`ceil`: https://numpy.org/doc/stable/reference/generated/numpy.ceil.html
.. _`trunc`: https://numpy.org/doc/stable/reference/generated/numpy.trunc.html
.. _`prod`: https://numpy.org/doc/stable/reference/generated/numpy.prod.html
.. _`sum`: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
.. _`nanprod`: https://numpy.org/doc/stable/reference/generated/numpy.nanprod.html
.. _`nansum`: https://numpy.org/doc/stable/reference/generated/numpy.nansum.html
.. _`cumprod`: https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html
.. _`cumsum`: https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
.. _`nancumprod`: https://numpy.org/doc/stable/reference/generated/numpy.nancumprod.html
.. _`nancumsum`: https://numpy.org/doc/stable/reference/generated/numpy.nancumsum.html
.. _`diff`: https://numpy.org/doc/stable/reference/generated/numpy.diff.html
.. _`ediff1d`: https://numpy.org/doc/stable/reference/generated/numpy.ediff1d.html
.. _`gradient`: https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
.. _`cross`: https://numpy.org/doc/stable/reference/generated/numpy.cross.html
.. _`trapz`: https://numpy.org/doc/stable/reference/generated/numpy.trapz.html
.. _`exp`: https://numpy.org/doc/stable/reference/generated/numpy.exp.html
.. _`expm1`: https://numpy.org/doc/stable/reference/generated/numpy.expm1.html
.. _`exp2`: https://numpy.org/doc/stable/reference/generated/numpy.exp2.html
.. _`log`: https://numpy.org/doc/stable/reference/generated/numpy.log.html
.. _`log10`: https://numpy.org/doc/stable/reference/generated/numpy.log10.html
.. _`log2`: https://numpy.org/doc/stable/reference/generated/numpy.log2.html
.. _`log1p`: https://numpy.org/doc/stable/reference/generated/numpy.log1p.html
.. _`logaddexp`: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
.. _`logaddexp2`: https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html
.. _`i0`: https://numpy.org/doc/stable/reference/generated/numpy.i0.html
.. _`sinc`: https://numpy.org/doc/stable/reference/generated/numpy.sinc.html
.. _`signbit`: https://numpy.org/doc/stable/reference/generated/numpy.signbit.html
.. _`copysign`: https://numpy.org/doc/stable/reference/generated/numpy.copysign.html
.. _`frexp`: https://numpy.org/doc/stable/reference/generated/numpy.frexp.html
.. _`ldexp`: https://numpy.org/doc/stable/reference/generated/numpy.ldexp.html
.. _`nextafter`: https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html
.. _`spacing`: https://numpy.org/doc/stable/reference/generated/numpy.spacing.html
.. _`lcm`: https://numpy.org/doc/stable/reference/generated/numpy.lcm.html
.. _`gcd`: https://numpy.org/doc/stable/reference/generated/numpy.gcd.html
.. _`add`: https://numpy.org/doc/stable/reference/generated/numpy.add.html
.. _`reciprocal`: https://numpy.org/doc/stable/reference/generated/numpy.reciprocal.html
.. _`positive`: https://numpy.org/doc/stable/reference/generated/numpy.positive.html
.. _`negative`: https://numpy.org/doc/stable/reference/generated/numpy.negative.html
.. _`multiply`: https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
.. _`divide`: https://numpy.org/doc/stable/reference/generated/numpy.divide.html
.. _`power`: https://numpy.org/doc/stable/reference/generated/numpy.power.html
.. _`subtract`: https://numpy.org/doc/stable/reference/generated/numpy.subtract.html
.. _`true_divide`: https://numpy.org/doc/stable/reference/generated/numpy.true_divide.html
.. _`floor_divide`: https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html
.. _`float_power`: https://numpy.org/doc/stable/reference/generated/numpy.float_power.html
.. _`fmod`: https://numpy.org/doc/stable/reference/generated/numpy.fmod.html
.. _`mod`: https://numpy.org/doc/stable/reference/generated/numpy.mod.html
.. _`modf`: https://numpy.org/doc/stable/reference/generated/numpy.modf.html
.. _`remainder`: https://numpy.org/doc/stable/reference/generated/numpy.remainder.html
.. _`divmod`: https://numpy.org/doc/stable/reference/generated/numpy.divmod.html
.. _`angle`: https://numpy.org/doc/stable/reference/generated/numpy.angle.html
.. _`real`: https://numpy.org/doc/stable/reference/generated/numpy.real.html
.. _`imag`: https://numpy.org/doc/stable/reference/generated/numpy.imag.html
.. _`conj`: https://numpy.org/doc/stable/reference/generated/numpy.conj.html
.. _`conjugate`: https://numpy.org/doc/stable/reference/generated/numpy.conjugate.html
.. _`maximum`: https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
.. _`fmax`: https://numpy.org/doc/stable/reference/generated/numpy.fmax.html
.. _`amax`: https://numpy.org/doc/stable/reference/generated/numpy.amax.html
.. _`nanmax`: https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html
.. _`minimum`: https://numpy.org/doc/stable/reference/generated/numpy.minimum.html
.. _`fmin`: https://numpy.org/doc/stable/reference/generated/numpy.fmin.html
.. _`amin`: https://numpy.org/doc/stable/reference/generated/numpy.amin.html
.. _`nanmin`: https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html
.. _`convolve`: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
.. _`clip`: https://numpy.org/doc/stable/reference/generated/numpy.clip.html
.. _`sqrt`: https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
.. _`cbrt`: https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html
.. _`square`: https://numpy.org/doc/stable/reference/generated/numpy.square.html
.. _`absolute`: https://numpy.org/doc/stable/reference/generated/numpy.absolute.html
.. _`fabs`: https://numpy.org/doc/stable/reference/generated/numpy.fabs.html
.. _`sign`: https://numpy.org/doc/stable/reference/generated/numpy.sign.html
.. _`heaviside`: https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html
.. _`nan_to_num`: https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html
.. _`real_if_close`: https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html
.. _`interp`: https://numpy.org/doc/stable/reference/generated/numpy.interp.html
