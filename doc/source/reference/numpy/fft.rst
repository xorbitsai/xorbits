Discrete Fourier Transform
==========================

The following table lists both implemented and not implemented methods. If you have need
of an operation that is listed as not implemented, feel free to open an issue on the
`GitHub repository`_, or give a thumbs up to already created issues. Contributions are
also welcome!

The following table is structured as follows: The first column contains the method name.
The second column contains link to a description of corresponding numpy method.
The third column is a flag for whether or not there is an implementation in Xorbits
for the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` standsfor partial 
(meaning some parameters may not be supported yet), and ``D`` stands for default to numpy.

Standard FFTs
-------------

+-----------------------+---------------+------------------------+----------------------------------+
| ``xorbits.numpy.fft`` | ``numpy.fft`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-----------------------+---------------+------------------------+----------------------------------+
| ``fft``               | `fft`_        | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``ifft``              | `ifft`_       | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``fft2``              | `fft2`_       | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``ifft2``             | `ifft2`_      | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``fftn``              | `fftn`_       | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``ifftn``             | `ifftn`_      | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+

Real FFTs
---------

+-----------------------+---------------+------------------------+----------------------------------+
| ``xorbits.numpy.fft`` | ``numpy.fft`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-----------------------+---------------+------------------------+----------------------------------+
| ``rfft``              | `rfft`_       | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``irfft``             | `irfft`_      | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``rfft2``             | `rfft2`_      | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``irfft2``            | `irfft2`_     | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``rfftn``             | `rfftn`_      | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``irfftn``            | `irfftn`_     | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+

Hermitian FFTs
--------------

+-----------------------+---------------+------------------------+----------------------------------+
| ``xorbits.numpy.fft`` | ``numpy.fft`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-----------------------+---------------+------------------------+----------------------------------+
| ``hfft``              | `hfft`_       | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``ihfft``             | `ihfft`_      | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+

Helper routines
---------------

+-----------------------+---------------+------------------------+----------------------------------+
| ``xorbits.numpy.fft`` | ``numpy.fft`` | Implemented? (Y/N/P/D) | Notes for Current implementation |
+-----------------------+---------------+------------------------+----------------------------------+
| ``fftfreq``           | `fftfreq`_    | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``rfftfreq``          | `rfftfreq`_   | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``fftshift``          | `fftshift`_   | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+
| ``ifftshift``         | `ifftshift`_  | Y                      |                                  |
+-----------------------+---------------+------------------------+----------------------------------+

.. _`GitHub repository`: https://github.com/xorbitsai/xorbits/issues
.. _`fft`: https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html
.. _`ifft`: https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html
.. _`fft2`: https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
.. _`ifft2`: https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html
.. _`fftn`: https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html
.. _`ifftn`: https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftn.html
.. _`rfft`: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
.. _`irfft`: https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html
.. _`rfft2`: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft2.html
.. _`irfft2`: https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html
.. _`rfftn`: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftn.html
.. _`irfftn`: https://numpy.org/doc/stable/reference/generated/numpy.fft.irfftn.html
.. _`hfft`: https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html
.. _`ihfft`: https://numpy.org/doc/stable/reference/generated/numpy.fft.ihfft.html
.. _`fftfreq`: https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html
.. _`rfftfreq`: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html
.. _`fftshift`: https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html
.. _`ifftshift`: https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html
