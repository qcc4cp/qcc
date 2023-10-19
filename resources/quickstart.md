# Quickstart Guide for "Quantum Computing for Programmers"

This quick start guide may help you getting started using this
code base by going through a few core concepts and
function calls.

#### Setup
This guide assumes you were successful in
downloading the Python sources from github
(`git clone https://github.com/qcc4cp/qcc.git`)
and installing the Python dependencies
`absl-py`, `numpy`, and `scipy` (for example, with `sudo pip install absl-py`).

For this guide, you don't need `bazel` or
compile the accelerated C++ routines. Make sure you 
point Python to the sources by setting the enviroment
variable `PYTHONPATH` to the root directory (`qcc`) of the sources.
For example, on Linux:
```
  export PYTHONPATH=/Users/rhundt/qcc
```
On Windows:
```
  # cmd.exe
  set  PYTHONPATH = C:\Users\rhundt\qcc
  
  # Powershell
  $Env:PYTHONPATH = C:\users\rhundt\qcc
```

Note that the main installation instruction use `bazel` to run the algorithms. This is not strictly necessary, you can run the algorithms individually just by invoking them on the Python command-line, such as:
```
   $ cd qcc/src
   $ python ./arith_classic.py   # and any of the other Python algorithms
   $ ...
```
   
#### First Steps
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

The second qubit remained in state `|0>`, so the probability of finding it in state `|1>` is zero. We can also simulate an actual measurement with or without state collapse. In this example we measure qubit 0 and force it to be measured as `|1>` (it is currently in superposition because of the prior Hadamard gate). We also want the state to collapse (and renormalize):
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

#### Bell State
Let's extend the previous example a litte bit and add a Controlled-Not gate from qubit 0 to qubit 1:
```
>>> from src.lib import state
>>> from src.lib import ops
>>> psi = state.bitstring(0, 0)
>>> op = ops.Hadamard() * ops.Identity()
>>> psi = op(psi)
>>> psi = ops.Cnot(0, 1)(psi)
>>> psi
State([0.70710677+0.j 0.        +0.j 0.        +0.j 0.70710677+0.j])
>>> psi.prob(0, 0)
0.49999997
>>> psi.prob(1, 0)
0.0
>>> psi.prob(0, 1)
0.0
>>> psi.prob(1, 1)
0.49999997
```
We created the entangled Bell state `1/sqrt(2) [1 0 0 1]^T`!

#### Classical Adder  
Now let's build something more complex - a classical 1-bit adder, constructed with qubits and quantum gates. As described in the book, the quantum circuit has several Controlled X-gates and also Controlled-Controlled X-gates!

```
a    ----o-----o---o-------
b    ----|--o--o---|--o----
cin  ----|--|--|---o--o--o-
sum  ----X--X--|---|--|--X-
cout ----------X---X--X----
```

We define a routine to apply the gates to a given state. Note how we use `Cnot` as well as `ControlledU` to further control some of the `Cnot` gates:
```
def fulladder_matrix(psi: state.State):
  psi = ops.Cnot(0, 3)(psi, 0)
  psi = ops.Cnot(1, 3)(psi, 1)
  psi = ops.ControlledU(0, 1, ops.Cnot(1, 4))(psi, 0)
  psi = ops.ControlledU(0, 2, ops.Cnot(2, 4))(psi, 0)
  psi = ops.ControlledU(1, 2, ops.Cnot(2, 4))(psi, 1)
  psi = ops.Cnot(2, 3)(psi, 2)
  return psi
```

To test this circuit, we can create a state with the corresponding inputs for `a`, `b`, `cin`, as well as `|0>` for `sum` and `cout`. For example, for a=0, b=1, and the carry-in = 1, the sum will be 1 + 1 = 0 mod 2, and the carry-out should be 1:
```
# a = 0, b = 1, cin = 1
psi = state.bitstring(0, 1, 1, 0, 0)
psi = fulladder_matrix(psi)

bsum, _ = ops.Measure(psi, 3, tostate=1, collapse=False)
bout, _ = ops.Measure(psi, 4, tostate=1, collapse=False)

>>> bsum
0.0
>>> bout
1.0
```


#### Conclusion
This concludes this short quick start guide, I hope it was useful. Please let me know if you wanted to see other or additional content here (qcc4cp@gmail.com).
