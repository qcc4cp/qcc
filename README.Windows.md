These instructions may be helpful for Windows.

Currently, Windows is partially supported:
*   You can run all algorithms and tests
*   `blaze test ...` does not work currently
*   The C++ accelerated library `libxgates` is currently not compiled to a DLL. All code runs via Python. This is typically not a problem, except for Shor's algorithm, which will run successfully, but very slow.

You have to ensure that you have installed `bazel` and `Python`. With `Python`, you need the packages (all installable via `pip install <package-name>`):
*   absl-py
*   numpy
*   scipy

