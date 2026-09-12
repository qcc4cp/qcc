These instructions may be helpful for Windows, which is currently only _partially_ supported:
*   You **can** run **all** algorithms and tests via Python.
*   The C++ accelerated library `libxgates` is currently **not** compiled to a DLL. Hence all code runs via Python, which is typically not a problem, except for Shor's algorithm (which will run very slowly).

Install `Python` [(installation instructions)](https://www.python.org/downloads/). You need the following packages, which can all be installed via `pip install <package-name>`:
*   absl-py
*   numpy
*   scipy

Because the accelerator is not built on Windows, there is nothing to configure
for Bazel — the algorithms run directly with Python via the automatic
pure-Python fallback.

Point the environment variable `PYTHONPATH` to the repository root so that the
`src` package is importable. For example, for `cmd.exe`:
```
set PYTHONPATH=C:\Users\robert_hundt\qcc
```
for Powershell:
```
$Env:PYTHONPATH = "C:\Users\robert_hundt\qcc"
```

With this, run any algorithm as a module from the repository root (note the
`src.` prefix and no `.py` extension):
```
C:\Users\robert_hundt\qcc> python -m src.deutsch
```
Run the library unit tests the same way, for example:
```
C:\Users\robert_hundt\qcc> python -m src.lib.bell_test
```
