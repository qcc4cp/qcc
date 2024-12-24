# Building libxgates.so

The book described how to accelerate Python with a C++ library and this
document described how to build this library.

#### Ingredients
The main source file for the library is in `src/lib/xgates.cc`. It has dependencies on Python headers, the Python library, 
and the numpy headers. 

To find the Python headers, you can run
```
python3 -c 'import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())'
```

To find the numpy headers, you can run
```
python3 -c 'import numpy; print(numpy.get_include())'`
```

The Python library will be somewhere in the neighborhood of these directories or in standard
Linux directories, eg
```
LIB=/usr/lib/python3.11/config-3.11-x86_64-linux-gnu/libpython3.11.so
```

Once these are found, you can build the library manually. All these steps are
in a script called [`qcc/make_libxgates.sh`](../make_libxgates.sh). It is recommended to just modify
this script to build the library on your system. 

The script determines the compiler option to build a loadable module (eg., `-shared`) and 
calls the compiler to build `qcc/libxgates.so`. For example:
```
OUT=./libxgates.so
cc -I${NUMPY} -I${PY} ${LIB} -O3 -ffast-math -DNPY_NO_DEPRECATED_API \
   -fPIC -std=c++0x ${SHARED} -o ${OUT} \
   src/lib/xgates.cc || exit 1
```

This builds the library in the root directory, which means you have to point the environment variable
`PYTHONPATH` to this directory (which you have to do anyways in order to import the other
Python modules).
