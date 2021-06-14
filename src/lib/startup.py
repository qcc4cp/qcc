# You can use this file to initialize an interactive python repl.
# For example, set (and adjust to your system):
#
#  export PYTHONPATH=$HOME/qcc/src/lib:$HOME/qcc:$HOME/qcc/bazel-bin/src/lib
#
#  export PYTHONSTARTUP=$HOME/qcc/src/lib/startup.py
#
#  python3
#  >> state.zeros(2)
#  State([1.+0.j 0.+0.j 0.+0.j 0.+0.j])

import bell
import circuit
import helper
import ops
import state
import tensor
print('startup.py: qcc initialized')
