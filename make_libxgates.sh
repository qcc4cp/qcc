#!/usr/bin/env bash
#
# Build the accelerated C++ extension src/lib/libxgates.so directly with the
# compiler. This is the recommended way to build the library; it does not
# require Bazel.
#
# All include paths are queried from the active Python interpreter, so the
# script adapts to whatever Python/NumPy is on your PATH (or in a virtualenv).
# It has been tested on macOS and Linux.
#
# Usage:
#     ./make_libxgates.sh
#
# To build against a specific interpreter (e.g. a virtualenv), set PYTHON:
#     PYTHON=/path/to/venv/bin/python ./make_libxgates.sh
#
# After building, make the library importable by adding src/lib to PYTHONPATH:
#     export PYTHONPATH=$PWD/src/lib
#
set -euo pipefail

# Locate the repository root (the directory this script lives in).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Choose the interpreter: $PYTHON if set, else a venv sibling, else python3.
PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    PYTHON="$(command -v python3)"
fi
echo "Python : ${PYTHON}"

# Query include directories from the interpreter (no hardcoded versions).
NUMPY_INC="$("${PYTHON}" -c 'import numpy; print(numpy.get_include())')"
PY_INC="$("${PYTHON}" -c "import sysconfig; print(sysconfig.get_path('include'))")"
echo "numpy  : ${NUMPY_INC}"
echo "python : ${PY_INC}"

# Shared-library flags per OS. Python symbols are resolved at load time from
# the embedding interpreter, so we do NOT link libpython directly. On macOS
# this needs -undefined dynamic_lookup; on Linux the default already allows
# unresolved symbols in a shared object.
#
# Parallelism (XGATES_PARALLEL env var, see xgates.cc):
#   * macOS uses Grand Central Dispatch, which lives in libSystem and needs
#     no extra flag or library.
#   * Linux uses OpenMP; -fopenmp both activates the code path (defines
#     _OPENMP) and links the runtime (libgomp), so it goes in SHARED.
OS="$(uname -s)"
case "${OS}" in
    Darwin) SHARED=(-dynamiclib -undefined dynamic_lookup) ;;
    Linux)  SHARED=(-shared -fopenmp) ;;
    *)
        echo "WARNING: unrecognized OS '${OS}'; assuming -shared." >&2
        SHARED=(-shared)
        ;;
esac
echo "OS     : ${OS}"

OUT="${REPO_ROOT}/src/lib/libxgates.so"
echo "Target : ${OUT}"

# Main compiler invocation. NPY_NO_DEPRECATED_API opts in to the modern
# (NumPy >= 1.7) C-API, which is required for NumPy 2.x.
cc -I"${NUMPY_INC}" -I"${PY_INC}" \
   -O3 -ffast-math -DNPY_NO_DEPRECATED_API \
   -fPIC -std=c++14 "${SHARED[@]}" \
   -o "${OUT}" \
   "${REPO_ROOT}/src/lib/xgates.cc"

echo "Made   :"
ls -l "${OUT}"
