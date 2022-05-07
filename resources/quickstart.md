# Quick Start Guide for "Quantum Computing for Programmers"

This quick start guide may help you getting started on this
code base by going through just a few core concepts and
function calls.

This guide assumes you were successful in
downloading the Python sources from github and installing the
Python dependencies, such as `absl-py` and `numpy`. No need for `blaze` or
compilation of the accelerated C++ routines. Make sure you 
point Python to the sources by setting the enviroment
variable `PYTHONPATH` to the root directory (`qcc`) of the sources.
For example (Linux):
```
  export PYTHONPATH=/Users/rhundt/qcc
```
Windows (cmd.exe)
```
  set  PYTHONPATH = C:\Users\rhundt\qcc
```

Let's start Python from the root directory - we will always `$` as the shell command-prompt and `>>>` as the Python
prompt. Note that your Python version may be different, it shouldn't matter:
```
/Users/rhundt/qcc $ python
Python 3.8.2 (default, Dec 21 2020, 15:06:04) 
[Clang 12.0.0 (clang-1200.0.32.29)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

As a first step, let's import the `Tensor` class which is used to 
represent all matrices, vectors, operators, and states:
```
>>> from src.lib import tensor
>>>
```

There is no error or warning message, which means this worked. We can double check:
```
>>> print(tensor)
<module 'src.lib.tensor' from '/Users/rhundt/qcc/src/lib/tensor.py'>
>>> 
```

Let's create a simple matrix with this class. Since `Tensor` is derived from the `numpy` `ndarray` data structure, it behaves just like a numpy array:
```
>>> A = tensor.Tensor([[1.0, 1.0], [1.0, -1.0]])
>>> A
Tensor([[ 1.+0.j,  1.+0.j],
        [ 1.+0.j, -1.+0.j]], dtype=complex64)
>>> 
```

Of course, we don't want to operate on simple tensors. Let's import our `State` class next:
```
>>> from src.lib import state
>>> 
```
This class provides a few functions to produce a state. A single qubit represents a state of, well, a single qubit. Let's create an initial qubit with probability amplitude for `|0>` being 0.5:
```
>>> q = state.qubit(alpha=0.5)
>>> q
State([0.5      +0.j 0.8660254+0.j])
```

Wasn't that easy? Check the file `src/lib/state.py` for other function to create a state. For example, to create the state `|1011>` you can simply call:
```
>>> psi = state.bitstring(1, 0, 1, 1)
>>> psi
State([0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j
       0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j])
```

Next we want to transform states with help of operators. We import the `Operator` class:
```
>>> from src.lib import ops
>>> 
```

The `Operator` class has several methods defining operators. For example, to construct a Hadamard gate you would call:
```
>>> H = ops.Hadamard()
>>> H
Operator([[ 0.70710677+0.j  0.70710677+0.j]
          [ 0.70710677+0.j -0.70710677+0.j]])
```

Let's create a state psi of 2 qubits corresponding to `|00>` and apply the Hadamard gate to qubit 0 (the leftmost qubit in state notation):
```
>>> psi = state.bitstring(0, 0)
>>> psi
State([1.+0.j 0.+0.j 0.+0.j 0.+0.j])
>>> op = ops.Hadamard() * ops.Identity()
>>> psi = op(psi)
>>> psi
State([0.70710677+0.j 0.        +0.j 0.70710677+0.j 0.        +0.j])
```

Note how we used the Python `*` operator to create the tensor product of the Hadamard and the identity gate. For plain matrix multiplication, the Python operator `@` is being used. Check out the file `ops.py` and discover the many other operators that are being defined in this file.

Given this state `psi`, what is now the probability of measureing `|00>`? We can used the `prob()` method of the state class to compute the probability from the probability amplitude of a state:
```
>>> psi.prob(0, 0)
0.49999997
>>> psi.prob(1, 0)
0.49999997
>>> psi.prob(1, 1)
0.0
>>> psi.prob(0, 1)
0.0
```

The second qubit remained in state `|0>`, so the probability of finding it in state `|1>` is zero. We can also similate an actual measurement with or without state collapse. In this example we measure qubit 0 and force it to be measured as `|1>` (it is currently in superposition because of the prior Hadamard gate). We also want the state to collapse (and renormalize):
```
>>> prob, new_state = ops.Measure(psi, idx=0, tostate=1, collapse=True)
>>> prob
0.49999997
>>> new_state
State([0.        +0.j 0.        +0.j 0.99999994+0.j 0.        +0.j])
>>> new_state.prob(1, 0)
0.9999999
```

The state is now `|10>` with close to 1.0 probability.

This concludes this short quick start guide, I hope it was useful. Please let me know if you wanted to see other or additional content here (qcc4cp@gmail.com).
