#!/usr/bin/env bash
# Run all algorithm .py targets in this directory.
#
# The algorithms import via "from src.lib import ...", so they must be run as
# modules (python -m src.<name>) from the repo root, with src/lib on
# PYTHONPATH so that "import libxgates" resolves to the compiled extension.
#
# This script does not use Bazel. On first run it builds the accelerated
# library src/lib/libxgates.so via ../make_libxgates.sh. All code also runs
# without the library, just about 10x+ slower.

set -u

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SRC_DIR}/.." && pwd)"

# Prefer the workspace venv interpreter; fall back to python3 on PATH. Export
# PYTHON so make_libxgates.sh uses the same interpreter.
PY="${REPO_ROOT}/../.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
    PY="$(command -v python3)"
fi
export PYTHON="${PY}"
echo "Python : ${PY}"

# Make "import libxgates" work; harmless if the .so is absent.
export PYTHONPATH="${SRC_DIR}/lib${PYTHONPATH:+:${PYTHONPATH}}"

# Build the accelerated library on first run (best effort). Algorithms still
# run via the Python fallback if the build fails.
if [[ ! -f "${SRC_DIR}/lib/libxgates.so" ]]; then
    echo "Building accelerated library (src/lib/libxgates.so) ..."
    if ! "${REPO_ROOT}/make_libxgates.sh"; then
        echo "*** NOTE: could not build libxgates.so.                    ***"
        echo "*** Algorithms will use the slower Python fallback.        ***"
    fi
fi

#
# Iterate over all algorithm files (sorted), skipping unit tests, and run
# each as a module from the repo root.
#
for path in $(ls -1 "${SRC_DIR}"/*.py | sort); do
    base="$(basename "${path}" .py)"
    [[ "${base}" == *_test ]] && continue
    echo
    echo "--- [${base}.py] ------------------------"
    ( cd "${REPO_ROOT}" && "${PY}" -m "src.${base}" ) || exit 1
done
