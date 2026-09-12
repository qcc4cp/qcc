// Python extension to accelerate gate applications.
//
#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <complex>

#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

// Phase-1 parallelization.
//
// The state-vector kernels below are "embarrassingly parallel" within a
// single gate application: a gate touches 2^(nbits-1) *independent* index
// pairs (butterflies), each reading and writing a disjoint pair of
// amplitudes psi[i] and psi[i+q2]. There are no cross-pair data
// dependencies, so the pairs can be distributed across threads with no
// locking. (Gate-to-gate ordering is still sequential; that dependency is
// handled by the Python layer applying gates one at a time.)
//
// Threading is controlled entirely at runtime by the XGATES_PARALLEL
// environment variable:
//   unset / "" / "0" / "1"  -> serial (original behavior, no overhead)
//   "N" (N > 1)             -> use up to N parallel work chunks
//
// On macOS we use Grand Central Dispatch (dispatch_apply), which is part of
// the base system and needs no extra libraries. On other platforms we fall
// back to OpenMP if the compiler was built with it, else serial.
#if defined(__APPLE__)
#define XGATES_USE_GCD 1
#include <dispatch/dispatch.h>
#elif defined(_OPENMP)
#define XGATES_USE_OMP 1
#include <omp.h>
#endif

typedef std::complex<double> cmplxd;
typedef std::complex<float> cmplxf;

// Read the desired parallelism from XGATES_PARALLEL exactly once and cache
// it. Returns the number of work chunks to split a gate into; a value <= 1
// means "run serially". Below a size threshold the caller ignores this and
// stays serial regardless, because thread-dispatch overhead dominates for
// small state vectors.
static int xgates_num_chunks() {
  static int cached = -1;
  if (cached < 0) {
    const char *env = getenv("XGATES_PARALLEL");
    int n = (env && *env) ? atoi(env) : 1;
    if (n < 1) n = 1;
    cached = n;
  }
  return cached;
}

// Only parallelize when there are at least this many amplitudes. Below this,
// a single thread already runs in tens of microseconds and dispatch overhead
// would make threading a net loss. 1<<16 == 65536 amplitudes.
static const long XGATES_PARALLEL_THRESHOLD = 1L << 16;

// apply1 applies a single gate to a state.
//
// Gates are typically 2x2 matrices, but in this implementation they
// are flattened to a 1x4 array:
//   |  a  b |
//   |  c  d |  -> | a b c d |
//
// The 2^(nbits-1) independent butterflies are addressed through a single
// linear index k. For target stride q2 = 2^tgt, butterfly k operates on
//   i       = ((k >> tgt) << (tgt + 1)) | (k & (q2 - 1))
//   partner = i + q2
// which is exactly the (g, i) double loop of the original code flattened
// into one iteration space. This uniform index space lets us split the work
// into equal chunks regardless of which qubit is targeted.
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
  const cmplx_type g0 = gate[0], g1 = gate[1], g2 = gate[2], g3 = gate[3];
  const long npairs = 1L << (nbits - 1);
  const long mask = q2 - 1;

  int chunks = xgates_num_chunks();
  if (chunks <= 1 || (1L << nbits) < XGATES_PARALLEL_THRESHOLD) {
    for (long k = 0; k < npairs; ++k) {
      long i = ((k >> tgt) << (tgt + 1)) | (k & mask);
      cmplx_type t1 = g0 * psi[i] + g1 * psi[i + q2];
      cmplx_type t2 = g2 * psi[i] + g3 * psi[i + q2];
      psi[i] = t1;
      psi[i + q2] = t2;
    }
    return;
  }
  if ((long)chunks > npairs) chunks = (int)npairs;

#if defined(XGATES_USE_GCD)
  const long per = (npairs + chunks - 1) / chunks;
  dispatch_apply(chunks, DISPATCH_APPLY_AUTO, ^(size_t c) {
    long lo = (long)c * per;
    long hi = lo + per; if (hi > npairs) hi = npairs;
    for (long k = lo; k < hi; ++k) {
      long i = ((k >> tgt) << (tgt + 1)) | (k & mask);
      cmplx_type t1 = g0 * psi[i] + g1 * psi[i + q2];
      cmplx_type t2 = g2 * psi[i] + g3 * psi[i + q2];
      psi[i] = t1;
      psi[i + q2] = t2;
    }
  });
#elif defined(XGATES_USE_OMP)
  #pragma omp parallel for num_threads(chunks) schedule(static)
  for (long k = 0; k < npairs; ++k) {
    long i = ((k >> tgt) << (tgt + 1)) | (k & mask);
    cmplx_type t1 = g0 * psi[i] + g1 * psi[i + q2];
    cmplx_type t2 = g2 * psi[i] + g3 * psi[i + q2];
    psi[i] = t1;
    psi[i + q2] = t2;
  }
#else
  for (long k = 0; k < npairs; ++k) {
    long i = ((k >> tgt) << (tgt + 1)) | (k & mask);
    cmplx_type t1 = g0 * psi[i] + g1 * psi[i + q2];
    cmplx_type t2 = g2 * psi[i] + g3 * psi[i + q2];
    psi[i] = t1;
    psi[i + q2] = t2;
  }
#endif
}

// applyc applies a controlled gate to a state.
//
// Same butterfly parallelization as apply1, with an added control-qubit
// predicate: the butterfly is applied only when the amplitude index has the
// control bit set. The predicate is evaluated on the low member i of each
// pair (i and i+q2 differ only in the target bit, which is distinct from the
// control bit, so both share the same control-bit value).
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
  const cmplx_type g0 = gate[0], g1 = gate[1], g2 = gate[2], g3 = gate[3];
  const long npairs = 1L << (nbits - 1);
  const long mask = q2 - 1;
  const long ctl_mask = 1L << ctl;

  int chunks = xgates_num_chunks();
  if (chunks <= 1 || (1L << nbits) < XGATES_PARALLEL_THRESHOLD) {
    for (long k = 0; k < npairs; ++k) {
      long i = ((k >> tgt) << (tgt + 1)) | (k & mask);
      if (i & ctl_mask) {
        cmplx_type t1 = g0 * psi[i] + g1 * psi[i + q2];
        cmplx_type t2 = g2 * psi[i] + g3 * psi[i + q2];
        psi[i] = t1;
        psi[i + q2] = t2;
      }
    }
    return;
  }
  if ((long)chunks > npairs) chunks = (int)npairs;

#if defined(XGATES_USE_GCD)
  const long per = (npairs + chunks - 1) / chunks;
  dispatch_apply(chunks, DISPATCH_APPLY_AUTO, ^(size_t c) {
    long lo = (long)c * per;
    long hi = lo + per; if (hi > npairs) hi = npairs;
    for (long k = lo; k < hi; ++k) {
      long i = ((k >> tgt) << (tgt + 1)) | (k & mask);
      if (i & ctl_mask) {
        cmplx_type t1 = g0 * psi[i] + g1 * psi[i + q2];
        cmplx_type t2 = g2 * psi[i] + g3 * psi[i + q2];
        psi[i] = t1;
        psi[i + q2] = t2;
      }
    }
  });
#elif defined(XGATES_USE_OMP)
  #pragma omp parallel for num_threads(chunks) schedule(static)
  for (long k = 0; k < npairs; ++k) {
    long i = ((k >> tgt) << (tgt + 1)) | (k & mask);
    if (i & ctl_mask) {
      cmplx_type t1 = g0 * psi[i] + g1 * psi[i + q2];
      cmplx_type t2 = g2 * psi[i] + g3 * psi[i + q2];
      psi[i] = t1;
      psi[i + q2] = t2;
    }
  }
#else
  for (long k = 0; k < npairs; ++k) {
    long i = ((k >> tgt) << (tgt + 1)) | (k & mask);
    if (i & ctl_mask) {
      cmplx_type t1 = g0 * psi[i] + g1 * psi[i + q2];
      cmplx_type t2 = g2 * psi[i] + g3 * psi[i + q2];
      psi[i] = t1;
      psi[i + q2] = t2;
    }
  }
#endif
}

// ---------------------------------------------------------------
// Python wrapper functions to call above accelerators.

template <typename cmplx_type, int npy_type>
void apply1_python(PyObject *param_psi, PyObject *param_gate,
                   int nbits, int tgt) {
  PyArrayObject *psi_arr =
      (PyArrayObject*) PyArray_FROM_OTF(param_psi, npy_type, NPY_ARRAY_IN_ARRAY);
  cmplx_type *psi = ((cmplx_type *)PyArray_GETPTR1(psi_arr, 0));

  PyArrayObject *gate_arr =
    (PyArrayObject*) PyArray_FROM_OTF(param_gate, npy_type, NPY_ARRAY_IN_ARRAY);
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
  PyArrayObject *psi_arr =
      (PyArrayObject*) PyArray_FROM_OTF(param_psi, npy_type, NPY_ARRAY_IN_ARRAY);
  cmplx_type *psi = ((cmplx_type *)PyArray_GETPTR1(psi_arr, 0));

  PyArrayObject *gate_arr =
    (PyArrayObject*) PyArray_FROM_OTF(param_gate, npy_type, NPY_ARRAY_IN_ARRAY);
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
  // import_array() initializes the NumPy C-API. It expands to a return
  // statement on failure, so it must run before the module is created.
  import_array();
  return PyModule_Create(&xgates_definition);
}

// To accommodate different build environments,
// this one might be needed.
PyMODINIT_FUNC PyInit_libxgates(void) {
  return PyInit_xgates();
}
