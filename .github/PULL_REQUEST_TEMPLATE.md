<!--
Thank you for your contribution!
-->

## What do these changes do?

# Changed all instances of 'import numpy as np' to simply 'import numpy'. This is to reduce confusion with np and xnp often being used in the same code and to help improve readability. There was also cases where 'import xorbits.numpy' and and 'from .... import numpy' were aliased as 'np instead of importing as 'xnp'. This is a problem as they were labelled as 'xnp' in other files that numpy and xorbits.numpy. These misslabeled 'np' aliases were corrected to 'xnp' to match other files. All 'xnp' aliases were left as xnp for readability. The more obvious and consistent naming should reduce ambiguity and improve clarity. Additionally, a note was added on line 32 of the 'numpy.rst' file in 'Getting started' to explicitly layout the lack of np and the separation of numpy and xnp

## Related issue number

<!-- Are there any issues opened that will be resolved by merging this change? -->
Fixes #153

## Check code requirements

- [X] tests added / passed (if needed)
- [X] Ensure all linting tests pass
