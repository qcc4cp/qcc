#include "libq.h"

#include <math.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
  libq::qureg* q = libq::new_qureg(0, 2);

  libq::h(0, q);
  libq::cx(0, 1, q);
  libq::u1(1, M_PI / 8.0, q);
  printf(" # States: %d\n", q->size);
  libq::print_qureg(q);

  libq::delete_qureg(q);
  return EXIT_SUCCESS;
}
