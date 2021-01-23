#include <stdio.h>

#include <cmath>

#include "libq.h"

namespace libq {

void v(int target, qureg *reg) {
  static cmplx mv[4] = {cmplx(0.5, 0.5), cmplx(0.5, 0.5),
                        cmplx(0.5, -0.5), cmplx(0.5, 0.5)};
  for (int i = 0; i < reg->size; ++i) {
    libq_gate1(target, mv, reg);
  }
}

void x(int target, qureg *reg) {
  for (int i = 0; i < reg->size; ++i) {
    reg->bit_xor(i, target);
  }
}

void y(int target, qureg *reg) {
  for (int i = 0; i < reg->size; ++i) {
    reg->bit_xor(i, target);
    if (reg->bit_is_set(i, target))
      reg->amplitude[i] *= cmplx(0, 1.0);
    else
      reg->amplitude[i] *= cmplx(0, -1.0);
  }
}

void z(int target, qureg *reg) {
  for (int i = 0; i < reg->size; ++i) {
    if (reg->bit_is_set(i, target)) {
      reg->amplitude[i] *= -1;
    }
  }
}

void h(int target, qureg *reg) {
  static cmplx mh[4] = {sqrt(1.0/2), sqrt(1.0/2), sqrt(1.0/2),
                        -sqrt(1.0/2)};
  libq_gate1(target, mh, reg);
}


void yroot(int target, qureg *reg) {
  static cmplx mv[4] = {cmplx(0.5, 0.5), cmplx(-0.5, -0.5),
                        cmplx(0.5, 0.5), cmplx(0.5, 0.5)};
  for (int i = 0; i < reg->size; ++i) {
    libq_gate1(target, mv, reg);
  }
}

void walsh(int width, qureg *reg) {
  for (int i = 0; i < width; ++i) {
    h(i, reg);
  }
}

cmplx cexp(float phi) {
  return cmplx(std::cos(phi), 0.0) +
         cmplx(0.0, 1.0) * cmplx(std::sin(phi), 0.0);
}

void u1(int target, float gamma, qureg *reg) {
  cmplx z = cexp(gamma);
  for (int i = 0; i < reg->size; ++i) {
    if (reg->bit_is_set(i, target)) {
      reg->amplitude[i] *= z;
    }
  }
}

void t(int target, qureg *reg) {
  cmplx z = cexp(M_PI / 4.0);
  for (int i = 0; i < reg->size; ++i) {
    if (reg->bit_is_set(i, target)) {
      reg->amplitude[i] *= z;
    }
  }
}

void cu1(int control, int target, float gamma, qureg *reg) {
  cmplx z = cexp(gamma);
  for (int i = 0; i < reg->size; ++i) {
    if (reg->bit_is_set(i, control)) {
      if (reg->bit_is_set(i, target)) {
        reg->amplitude[i] *= z;
      }
    }
  }
}

void cv(int control, int target, qureg *reg) {
  for (int i = 0; i < reg->size; ++i) {
    if (reg->bit_is_set(i, control)) {
      if (reg->bit_is_set(i, target)) {
        static cmplx mv[4] = {cmplx(0.5, 0.5), cmplx(0.5, -0.5),
                              cmplx(0.5, -0.5), cmplx(0.5, 0.5)};
        libq_gate1(target, mv, reg);
      }
    }
  }
}

void cv_adj(int control, int target, qureg *reg) {
  for (int i = 0; i < reg->size; ++i) {
    if (reg->bit_is_set(i, control)) {
      if (reg->bit_is_set(i, target)) {
        static cmplx mv[4] = {cmplx(0.5, -0.5), cmplx(0.5, 0.5),
                              cmplx(0.5, 0.5), cmplx(0.5, -0.5)};
        libq_gate1(target, mv, reg);
      }
    }
  }
}

void cx(int control, int target, qureg *reg) {
  for (int i = 0; i < reg->size; ++i) {
    if (reg->bit_is_set(i, control)) {
      reg->bit_xor(i, target);
    }
  }
}

void cz(int control, int target, qureg *reg) {
  for (int i = 0; i < reg->size; ++i) {
    if (reg->bit_is_set(i, control)) {
      if (reg->bit_is_set(i, target)) {
        reg->amplitude[i] *= -1;
      }
    }
  }
}

void ccx(int control0, int control1, int target, qureg *reg) {
  for (int i = 0; i < reg->size; ++i) {
    if (reg->bit_is_set(i, control0)) {
      if (reg->bit_is_set(i, control1)) {
        reg->bit_xor(i, target);
      }
    }
  }
}

void flush(qureg* reg) {
  print_qureg_stats(reg);
}


}  // namespace libq
