These instructions may be helpful for running this code base in SageMath.

SageMath is based on Python and it is comparatively easy to run
the code in this project under SageMath, either via command-line
or interactively.

To run a particular algorithm on the command-line, simply run something like:

```
$ sage deutsch.py  # or any of the other algorithms.
```

To start an interactive SageMath session with all the code preloaded, including
the accelerator library (`xgates`), this may be a way to do it:
####   Prepare a file `startup.py`

Create a file `startup.py` (or any other name of your chosing). 
This file should import all the source files in the `src/lib` directory. 
For example:
```
"""Initialize and load all qcc packages."""

# pylint: disable=unused-import
import numpy as np

from src.lib import bell
from src.lib import circuit
from src.lib import helper
from src.lib import ops
from src.lib import state
from src.lib import tensor
print('QCC: ' + __file__ + ': initialized')
```

#### Set environment
Once the file is setup, point the `PYTHONSTARTUP` environment variable to it
(also set `PYTHONPATH` to find the accelerated `xgates` library). For example (with similar constructions on Windows):
```
export PYTHONPATH=$HOME/qcc:$HOME/qcc/bazel-bin/src/lib
export PYTHONSTARTUP=$HOME/qcc/src/lib/startup.py
```

#### Start SageMath
On startup, the file will be preloaded and SageMath should print something like this, ready for an interactive session.
```
$ sage
SageMath version 9.4, Release Date: 2021-08-22
QCC: /usr/local/google/home/rhundt/qcc/src/lib/startup.py: initialized
sage: psi = state.bitstring(1, 0, 1, 1)
sage: psi
State([0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j
       0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j])
sage: psi = ops.PauliX()(psi, 1)
sage: psi
State([0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j
       0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j])
sage: 
```
