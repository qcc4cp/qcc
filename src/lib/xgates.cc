// Python extension to accelerate gate applications.
//
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <complex>

#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>
#include <numpy/npy_3kcompat.h>

typedef std::complex<double> cmplxd;
typedef std::complex<float> cmplxf;

// apply1 applies a single gate to a state.
//
// Gates are typically 2x2 matrices, but in this implementation they
// are flattened to a 1x4 array:
//   |  a  b |
//   |  c  d |  -> | a b c d |
//
template <typename cmplx_type>
void apply1(cmplx_type *psi, cmplx_type gate[4],
            int nbits, int tgt) {
  tgt = nbits - tgt - 1;
  int q2 = 1 << tgt;
  if (q2 < 0) {
    fprintf(stderr, "***Error***: Negative qubit index in apply1().\n");
    fprintf(stderr, "             Perhaps using wrongly shaped state?\n");
    exit(EXIT_FAILURE);
  }
  for (int g = 0; g < 1 << nbits; g += (1 << (tgt+1))) {
    for (int i = g; i < g + q2; ++i) {
      cmplx_type t1 = gate[0] * psi[i] + gate[1] * psi[i + q2];
      cmplx_type t2 = gate[2] * psi[i] + gate[3] * psi[i + q2];
      psi[i] = t1;
      psi[i + q2] = t2;
    }
  }
}

// applyc applies a controlled gate to a state.
//
template <typename cmplx_type>
void applyc(cmplx_type *psi, cmplx_type gate[4],
            int nbits, int ctl, int tgt) {
  tgt = nbits - tgt - 1;
  ctl = nbits - ctl - 1;
  int q2 = 1 << tgt;
  if (q2 < 0) {
    fprintf(stderr, "***Error***: Negative qubit index in applyc().\n");
    fprintf(stderr, "             Perhaps using wrongly shaped state?\n");
    exit(EXIT_FAILURE);
  }
  for (int g = 0; g < 1 << nbits; g += 1 << (tgt+1)) {
    for (int i = g; i < g + q2; ++i) {
      int idx = g * (1 << nbits) + i;
      if (idx & (1 << ctl)) {
        cmplx_type t1 = gate[0] * psi[i] + gate[1] * psi[i + q2];
        cmplx_type t2 = gate[2] * psi[i] + gate[3] * psi[i + q2];
        psi[i] = t1;
        psi[i + q2] = t2;
      }
    }
  }
}

// ---------------------------------------------------------------
// Python wrapper functions to call above accelerators.

template <typename cmplx_type, int npy_type>
void apply1_python(PyObject *param_psi, PyObject *param_gate,
                   int nbits, int tgt) {
  PyObject *psi_arr =
    PyArray_FROM_OTF(param_psi, npy_type, NPY_IN_ARRAY);
  cmplx_type *psi = ((cmplx_type *)PyArray_GETPTR1(psi_arr, 0));

  PyObject *gate_arr =
    PyArray_FROM_OTF(param_gate, npy_type, NPY_IN_ARRAY);
  cmplx_type *gate = ((cmplx_type *)PyArray_GETPTR1(gate_arr, 0));

  apply1<cmplx_type>(psi, gate, nbits, tgt);

  Py_DECREF(psi_arr);
  Py_DECREF(gate_arr);
}

static PyObject *apply1_c(PyObject *dummy, PyObject *args) {
  PyObject *param_psi = NULL;
  PyObject *param_gate = NULL;
  int nbits;
  int tgt;
  int bit_width;

  if (!PyArg_ParseTuple(args, "OOiii", &param_psi, &param_gate,
                        &nbits, &tgt, &bit_width))
    return NULL;
  if (bit_width == 128) {
    apply1_python<cmplxd, NPY_CDOUBLE>(param_psi,
                                       param_gate, nbits, tgt);
  } else {
    apply1_python<cmplxf, NPY_CFLOAT>(param_psi,
                                      param_gate, nbits, tgt);
  }
  Py_RETURN_NONE;
}

template <typename cmplx_type, int npy_type>
void applyc_python(PyObject *param_psi, PyObject *param_gate,
                   int nbits, int ctl, int tgt) {
  PyObject *psi_arr =
    PyArray_FROM_OTF(param_psi, npy_type, NPY_IN_ARRAY);
  cmplx_type *psi = ((cmplx_type *)PyArray_GETPTR1(psi_arr, 0));

  PyObject *gate_arr =
    PyArray_FROM_OTF(param_gate, npy_type, NPY_IN_ARRAY);
  cmplx_type *gate = ((cmplx_type *)PyArray_GETPTR1(gate_arr, 0));

  applyc<cmplx_type>(psi, gate, nbits, ctl, tgt);

  Py_DECREF(psi_arr);
  Py_DECREF(gate_arr);
}

static PyObject *applyc_c(PyObject *dummy, PyObject *args) {
  PyObject *param_psi = NULL;
  PyObject *param_gate = NULL;
  int nbits;
  int ctl;
  int tgt;
  int bit_width;

  if (!PyArg_ParseTuple(args, "OOiiii", &param_psi, &param_gate,
                        &nbits, &ctl, &tgt, &bit_width))
    return NULL;
  if (bit_width == 128) {
    applyc_python<cmplxd, NPY_CDOUBLE>(param_psi,
                                       param_gate, nbits, ctl, tgt);
  } else {
    applyc_python<cmplxf, NPY_CFLOAT>(param_psi,
                                      param_gate, nbits, ctl, tgt);
  }
  Py_RETURN_NONE;
}

// ---------------------------------------------------------------
// Python boilerplate to expose above wrappers to programs.
//
static PyMethodDef xgates_methods[] = {
    {"apply1", apply1_c, METH_VARARGS,
     "Apply single-qubit gate, complex double"},
    {"applyc", applyc_c, METH_VARARGS,
     "Apply controlled qubit gate, complex double"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef xgates_definition = {
  PyModuleDef_HEAD_INIT,
  "xgates",
  "Python extension to accelerate quantum simulation math",
  -1,
  xgates_methods
};

PyMODINIT_FUNC PyInit_xgates(void) {
  Py_Initialize();
  import_array();
  return PyModule_Create(&xgates_definition);
}

// To accommodate different build environments,
// this one might be needed.
PyMODINIT_FUNC PyInit_libxgates(void) {
  return PyInit_xgates();
}
