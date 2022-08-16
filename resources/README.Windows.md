These instructions may be helpful for Windows, which is currently only _partially_ supported:
*   You **can** run **all** algorithms and tests
*   `blaze test ...` does **not** work currently
*   The C++ accelerated library `libxgates` is currently **not** compiled to a DLL. Hence all code runs via Pythonm which is typically not a problem, except for Shor's algorithm (which will run very slowly).

You have to ensure that you have installed `bazel` [(installation instructions)](https://bazel.build/install/windows) and `Python` [(installation instructions)](https://www.python.org/downloads/). With `Python`, you need the following packages, which can all be installed via `pip install <package-name>`:
*   absl-py
*   numpy
*   scipy

Edit the `WORKSPACE` file in the root directory and adjust the paths according to your installation and following Windows' path syntax. For example (for Robert's current installation - your directories may be different):
```
new_local_repository(
    name = "third_party_python",
    build_file = __workspace_dir__ + "/python.BUILD",
    # Configure:
    path = "C:\\Program Files\\Python37\\include"
)

new_local_repository(
    name = "third_party_numpy",
    build_file = __workspace_dir__ + "/numpy.BUILD",
    # Configure:
    path = "C:\\Users\\robert_hundt\\AppData\\Roaming\\Python\\Python37\\site-packages\\numpy\\core"
)
```

Finally, point the environment variable `PYTHONPATH` to the root directory. For example, for `cmd.exe`:
```
set PYTHONPATH = "C:\Users\robert_hundt\qcc"
```
for Powershell:
```
$Env:PYTHONPATH = "C:\Users\robert_hundt\qcc"
```

With this, you can run everything, for example:
```
qcc $  cd src
qcc/src $ bazel run deutsch  # and all other algos
qcc/src $ cd lib
qcc/src/lib $ bazel run bell_test   # and all other tests
```
