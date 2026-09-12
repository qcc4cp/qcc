# Building libxgates.so

The book describes how to accelerate Python with a C++ library. This
document describes how to build that library. Building it is optional: all
algorithms run without it, just about 10x slower, via a pure-Python fallback.

#### The easy way

Run the provided script from the repository root:
```
$ ./make_libxgates.sh
```
The script queries the active Python interpreter for the Python and NumPy
header locations, picks the right flags for your OS (macOS or Linux), and
builds `src/lib/libxgates.so`. It does not hardcode any interpreter version or
path and does not link `libpython` (Python symbols are resolved at load time
from the embedding interpreter).

To build against a specific interpreter, for example a virtualenv, set
`PYTHON`:
```
$ PYTHON=/path/to/venv/bin/python ./make_libxgates.sh
```

Once built, make the library importable by pointing `PYTHONPATH` at `src/lib`
(the same variable you set to import the other modules):
```
$ export PYTHONPATH=$PWD/src/lib
```

#### What the script does

The main source file is `src/lib/xgates.cc`. It depends on the Python C
headers and the NumPy C headers, discovered as:
```
# Python headers:
python3 -c "import sysconfig; print(sysconfig.get_path('include'))"

# NumPy headers:
python3 -c 'import numpy; print(numpy.get_include())'
```
(The old `distutils.sysconfig` API was removed in Python 3.12; `sysconfig` is
the modern replacement.)

The compiler invocation is essentially:
```
cc -I${NUMPY_INC} -I${PY_INC} \
   -O3 -ffast-math -DNPY_NO_DEPRECATED_API \
   -fPIC -std=c++14 ${SHARED} \
   -o src/lib/libxgates.so \
   src/lib/xgates.cc
```
where `${SHARED}` is `-dynamiclib -undefined dynamic_lookup` on macOS and
`-shared` on Linux. `NPY_NO_DEPRECATED_API` opts in to the modern NumPy C-API
and is required for NumPy 2.x.

#### Building with Bazel instead

Alternatively, build the library with Bazel:
```
$ bazel build //src/lib:libxgates.so
```
See [README.Linux.md](README.Linux.md) for the Bazel setup.
