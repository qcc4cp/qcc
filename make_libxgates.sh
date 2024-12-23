# Simple command line to build libxgates.so
#
# The directories and libraries need to be adjusted for a given setup.
# This has been tested on MacOS and Ubuntu.

#
# get numpy include directory:
#
NUMPY=`python3 -c 'import numpy;\
                   print(numpy.get_include())'`
echo "numpy  : ${NUMPY}"

#
# Directory where to find the Python.h file in:
#
PY=`python3 -c 'import distutils.sysconfig;\
                print(distutils.sysconfig.get_python_inc())'`
echo "Python : ${PY}"


#
# Python library. It can be hard to find on your system but may be close
# to one of the paths determined above.
#
# Example: MacOS / Darwin
#
LIB=/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11\
/./config-3.11-darwin/libpython3.11.dylib
#
# Example Linux / Ubuntu
#
# LIB=/usr/lib/python3.11/config-3.11-x86_64-linux-gnu/libpython3.11.so
#
echo "Library: ${LIB}"

#
# Command-line option to make shared module
#
SHARED=""
OS=`uname -a | awk '{print $1}'`
if [[ ${OS} == "Darwin" ]]; then
    SHARED="-dynamiclib"
fi
if [[ ${OS} == "Linux" ]]; then
    SHARED="-shared"
fi
if [[ ${SHARED} == "" ]]; then
    echo "WARNING: Could not recognize the OS ($OS)."
    echo "         Check flags to make shared object."
    exit 1
fi
echo "Flags  : ${SHARED}"

#
# Target
#
OUT=./libxgates.so
echo "Target : ${OUT}"

#
# Main compiler invokation:
#
cc -I${NUMPY} -I${PY} ${LIB} -O3 -ffast-math -DNPY_NO_DEPRECATED_API \
   -fPIC -std=c++0x -MD ${SHARED} \
   src/lib/xgates.cc -o ${OUT} || exit 1

echo "Made   :"
ls -l ${OUT}
