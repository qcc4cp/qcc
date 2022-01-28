#include <stdio.h>

#include "libq.h"

namespace libq {

float probability(cmplx ampl) {
  return ampl.real() * ampl.real() + ampl.imag() * ampl.imag();
}

qureg *new_qureg(state_t initval, int width) {
  qureg *reg = new qureg;

  reg->width = width;
  reg->size = 1;
  reg->maxsize = 0;
  reg->hash_computes = 0;
  reg->hashw = width + 2;

  /* Allocate memory for 1 base state */
  reg->state = static_cast<state_t *>(calloc(1, sizeof(state_t)));
  reg->amplitude = static_cast<cmplx *>(calloc(1, sizeof(cmplx)));

  /* Allocate the hash table */
  reg->hash = static_cast<int *>(calloc(1 << reg->hashw, sizeof(int)));

  // For libq_arith_test, this technique brings runtime down from 3.2 secs
  // to 1.2 secs! Having super efficient hash-tables + management will make
  // all the difference for sparse quantum circuits.
  //
  reg->hash_caching = true;
  reg->hash_hits = nullptr;
  reg->hits = 0;
  if (reg->hash_caching) {
    reg->hash_hits = static_cast<int *>(calloc(HASH_CACHE_SIZE, sizeof(int)));
  }

  /* Initialize the quantum register */
  reg->state[0] = initval;
  reg->amplitude[0] = 1;

  return reg;
}

void delete_qureg(qureg *reg) {
  if (reg->hash) {
    free(reg->hash);
    reg->hash = nullptr;
  }
  if (reg->hash_hits) {
    free(reg->hash_hits);
    reg->hash_hits = nullptr;
    reg->hits = 0;
  }
  free(reg->amplitude);
  reg->amplitude = nullptr;
  if (reg->state) {
    free(reg->state);
    reg->state = nullptr;
  }
  delete (reg);
}

void print_qureg(qureg *reg) {
  printf("States with non-zero probability:\n");
  for (int i = 0; i < reg->size; ++i) {
    printf("  % f %+fi|%llu> (%e) (|", reg->amplitude[i].real(),
           reg->amplitude[i].imag(), reg->state[i],
           probability(reg->amplitude[i]));
    for (int j = reg->width - 1; j >= 0; --j) {
      if (j % 4 == 3) {
        printf(" ");
      }
      printf("%i", (((static_cast<state_t>(1) << j) & reg->state[i]) > 0));
    }
    printf(">)\n");
  }
}

void print_qureg_stats(qureg *reg) {
  printf("# of qubits        : %d\n", reg->width);
  printf("# of hash computes : %d\n", reg->hash_computes);
  printf("Maximum # of states: %d, theoretical: %d, %.3f%%\n",
         reg->maxsize, 2 << reg->width,
         100.0 * reg->maxsize / (2 << reg->width));
}

}  // namespace libq
