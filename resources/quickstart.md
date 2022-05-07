# Quick Start Guide for: Quantum Computing for Programmers

This little quick start guide should help you get started on this
new code base by going through a few core concepts and
function calls.

This guide only assumes you were successful in
downloading the Python sources from github and installing the
Python dependencies, such as `absl-py` and `numpy`, and that
you point Python to the sources by setting the enviroment
variable `PYTHONPATH` to the root directory (`qcc`) of the sources.
For example (Linux):
```
  export PYTHONPATH=/Users/rhundt/qcc
```
Windows cmd.exe
```
  set  PYTHONPATH = C:\Users\rhundt\qcc
```

Let's start Python - we will always `$` as the shell command-prompt and `>>>` as the Python
prompt. Note that your python version may be different, it shouldn't matter:
```
$ python
Python 3.8.2 (default, Dec 21 2020, 15:06:04) 
[Clang 12.0.0 (clang-1200.0.32.29)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```
