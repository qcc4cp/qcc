#ifndef LIBQ_LIBQ_H_
#define LIBQ_LIBQ_H_

#include <complex>

// #include "base/integral_types.h"

namespace libq {

typedef std::complex<float> cmplx;
typedef unsigned long long state_t;
#define HASH_CACHE_SIZE (1024 * 64)

struct qureg_t {
  cmplx* amplitude;
  state_t* state;
  int width; /* number of qubits in the qureg */
  int size;  /* number of non-zero vectors */
  int maxsize; /* largest size reached */
  int hash_computes; /* how often is the hash table computed */
  int hashw; /* width of the hash array */
  int* hash;

  // Certain circuits have a very large number of theoretical states,
  // but only a very small number of states with non-zero probability.
  // For those inputs, it can be beneficial to not memset the whole
  // hash table, but only the elements that have been set.
  //
  // This can be enabled by setting hash_caching to true.
  //
  bool hash_caching; /* Cache hash values, optional */
  int* hash_hits;
  int  hits;

  bool bit_is_set(int index, int target) __attribute__ ((pure)) {
    return state[index] & (static_cast<state_t>(1) << target);
  }
  void bit_xor(int index, int target) {
    state[index] ^= (static_cast<state_t>(1) << target);
  }
};
typedef struct qureg_t qureg;

qureg *new_qureg(state_t initval, int width);
void delete_qureg(qureg *reg);
void print_qureg(qureg *reg);
void print_qureg_stats(qureg *reg);
void flush(qureg* reg);

void x(int target, qureg *reg);
void y(int target, qureg *reg);
void z(int target, qureg *reg);
void h(int target, qureg *reg);
void t(int target, qureg *reg);
void v(int target, qureg *reg);
void yroot(int target, qureg *reg);
void walsh(int width, qureg *reg);
void cx(int control, int target, qureg *reg);
void cz(int control, int target, qureg *reg);
void ccx(int control0, int control1, int target, qureg *reg);
void u1(int target, float gamma, qureg *reg);
void cu1(int control, int target, float gamma, qureg *reg);
void cv(int control, int target, qureg *reg);
void cv_adj(int control, int target, qureg *reg);

// -- Internal ----------------------------------------

float probability(cmplx ampl);
void libq_gate1(int target, cmplx m[4], qureg *reg);

}  // namespace libq

#endif  // LIBQ_LIBQ_H_
