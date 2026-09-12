# Installation on macOS

Most of the code is Python and runs out of the box once the Python
dependencies are installed. The optional C++ accelerator (`libxgates`) speeds
up simulation but is not required — there is an automatic pure-Python fallback.

## Dependencies

Install the Python packages (into a virtualenv or your system Python):
```
python3 -m pip install absl-py numpy scipy
```
A C/C++ compiler (`clang`, provided by the Xcode command-line tools) is needed
only to build the accelerator:
```
xcode-select --install
```
Get the sources:
```
git clone https://github.com/qcc4cp/qcc.git
```

## Build the accelerator (optional)

From the repository root:
```
./make_libxgates.sh
```
The script queries the active Python interpreter for the Python and NumPy
header paths and picks the correct macOS flags
(`-dynamiclib -undefined dynamic_lookup`) automatically — there is nothing to
configure and no interpreter version is hardcoded. To build against a specific
interpreter, e.g. a virtualenv:
```
PYTHON=/path/to/venv/bin/python ./make_libxgates.sh
```
This produces `src/lib/libxgates.so`. Make it importable:
```
export PYTHONPATH=$PWD/src/lib
```

Alternatively, build it with Bazel (Bazel 7+, Bzlmod):
```
bazel build //src/lib:libxgates.so
```
The Python and NumPy headers are discovered automatically by the module
extension in `bazel/python_headers.bzl`; it prefers a virtualenv at
`../.venv/bin/python`, or you can point it at any interpreter with
`--repo_env=PYTHON_BIN=/path/to/python`.

## Run

Without Bazel, run each algorithm as a module from the repository root (the
algorithms import `from src.lib import ...`):
```
export PYTHONPATH=$PWD/src/lib
python3 -m src.grover
```
Run them all with the helper script (it builds the accelerator on first use):
```
./src/runall.sh
```
With Bazel:
```
bazel test //src/lib:all      # run the library unit tests
bazel run  //src:grover       # run an algorithm
```
