# Manual Installation on Linux

The following instructions focus on Debian Linux but should work for Ubuntu as well. 
Note that if you can use Docker, many of these steps are performed for you by Docker when
the container is being created.

## Dependencies

To run the code a few tools are needed:

*  The use of the `bazel` build system is optional but can be helpful.\
   Install from [bazel's homepage](https://docs.bazel.build/versions/master/install.html)

*  We will need Python's `pip` tool to install packages and `git` to manage the source.
  Here is one way to install them:
```
    sudo apt-get install python3-pip
    sudo apt-get install git
```

*  We need Google's `absl` library, as well as `numpy` and `scipy`. Install with
```
   sudo python3 -m pip install absl-py
   sudo python3 -m pip install numpy
   sudo python3 -m pip install scipy
```

* Finally, to get these source onto your computer:
```
    git clone https://github.com/qcc4cp/qcc.git
```

## Build

Much of the code is in Python and will run out of the box. There is
some C++ for the high performance simulation (`libxgates`). *The Python code
runs without C++ acceleration, just much slower.*

There are two ways to build the accelerated library:

1.  **Recommended: the script [`make_libxgates.sh`](../make_libxgates.sh)**,
    documented [here](README.buildxgates.md). From the repository root:
    ```
    ./make_libxgates.sh
    ```
    It queries the active Python interpreter for the Python and NumPy header
    locations, so there is nothing to configure. To build against a specific
    interpreter (e.g. a virtualenv), set `PYTHON`:
    ```
    PYTHON=/path/to/venv/bin/python ./make_libxgates.sh
    ```

2.  **With `bazel`** (Bazel 7+, using Bzlmod):
    ```
    bazel build //src/lib:libxgates.so
    ```
    The build rule is in `src/lib/BUILD`. The C++ extension needs the Python
    (`Python.h`) and `numpy` C headers. These are located automatically by the
    module extension in `bazel/python_headers.bzl`, which queries an
    interpreter chosen as, in order: the `PYTHON_BIN` environment variable, a
    virtualenv at `../.venv/bin/python`, then `python3` on `PATH`. To force a
    specific interpreter:
    ```
    bazel build //src/lib:libxgates.so --repo_env=PYTHON_BIN=/path/to/python
    ```
    Third-party Python packages (`numpy`, `scipy`, `absl-py`) are resolved by
    Bzlmod from `requirements_lock.txt` (see `MODULE.bazel`); no system-wide
    pip install is required for the Bazel path.

Once built, `libxgates.so` is imported by `circuit.py` (as
`import libxgates as xgates`), with an automatic pure-Python fallback if it is
not found. Make it importable by adding its directory to `PYTHONPATH`:
```
export PYTHONPATH=$PWD/src/lib
```

## Run

To build the library and verify the installation with `bazel`:
```
    bazel build //src/lib:libxgates.so
    bazel test //src/lib:all
```

To run the individual algorithms with `bazel`:
```
    bazel run //src:grover        # and any other algorithm target
```

Or, without `bazel`, run each algorithm as a module from the repository root
(the algorithms import `from src.lib import ...`):
```
    export PYTHONPATH=$PWD/src/lib
    python3 -m src.grover
```

Or run them all at once with the helper script, which builds the accelerator
on first use:
```
    ./src/runall.sh
```

## Minimal Setup
If you can't get `libxgates` to build, you can still run all Python algorithms;
they just run more slowly. From the repository root:
```
  export PYTHONPATH=$PWD/src/lib
  python3 -m src.estimate_pi
```







