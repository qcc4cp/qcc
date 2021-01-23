// This file attempts to 'jit' or 'inline' gates for which this is
// possible. For something like order finding about 70% of all the
// gates can be combined, which could speed up execution significantly.
//
// The approach is to 'collect' fusable operations (which are all the
// ops that don't call libq_gate1) and then execute them sequentially
// on each state, thus reducing the number of iterations over all
// states.
//
// Performance experiments seem to show NO benefit. It is likely that
// vectorization is broken by the 'switch-per-op' approach. The
// benefits from a reduced number of state traversals is not
// materializing. We need to combined operations without the jumps
// between them, in order to allow vectorization. A different experiment
// will explore this options/
//
#include <stdio.h>

#include <cmath>
#include <vector>

#include "libq.h"

namespace libq {

  typedef enum { X, Y, Z, T, U1, CU1, CX, CZ, CCX } op_t;

class Op {
 public:
  Op(op_t op, int target) : op_(op), target_(target), ctl0_(0), ctl1_(0) {}
  Op(op_t op, int target, cmplx val)
      : op_(op), target_(target), ctl0_(0), ctl1_(0), val_(val) {}
  Op(op_t op, int ctl, int target)
      : op_(op), target_(target), ctl0_(ctl), ctl1_(0) {}
  Op(op_t op, int ctl0, int ctl1, int target)
      : op_(op), target_(target), ctl0_(ctl0), ctl1_(ctl1) {}
  Op(op_t op, int ctl, int target, cmplx val)
      : op_(op), target_(target), ctl0_(ctl), ctl1_(0), val_(val) {}

  op_t op() { return op_; }
  int target() { return target_; }
  int control() { return ctl0_; }
  int control0() { return ctl0_; }
  int control1() { return ctl1_; }
  cmplx val() { return val_; }

 private:
  op_t op_;
  int target_, ctl0_, ctl1_;
  cmplx val_;
};

class Jit {
 public:
  Jit() : combined_gates_(0) {}

  int combined_gates() { return combined_gates_; }

  void AddOp(Op operation) { op_list_.push_back(operation); }

  void Execute(qureg *reg) {
    for (int i = 0; i < reg->size; ++i) {
      for (auto op : op_list_) {
        switch (op.op()) {
          case op_t::X:
            reg->bit_xor(i, op.target());
            break;

          case op_t::Y:
            reg->bit_xor(i, op.target());
            if (reg->bit_is_set(i, op.target()))
              reg->amplitude[i] *= cmplx(0, 1.0);
            else
              reg->amplitude[i] *= cmplx(0, -1.0);
            break;

          case op_t::Z:
            if (reg->bit_is_set(i, op.target())) {
              reg->amplitude[i] *= -1;
            }
            break;

          case op_t::CX:
            if (reg->bit_is_set(i, op.control())) {
              reg->bit_xor(i, op.target());
            }
            break;

          case op_t::CZ:
            if (reg->bit_is_set(i, op.control())) {
              if (reg->bit_is_set(i, op.target())) {
                reg->amplitude[i] *= -1;
              }
            }
            break;
          case op_t::CCX:
            if (reg->bit_is_set(i, op.control0())) {
              if (reg->bit_is_set(i, op.control1())) {
                reg->bit_xor(i, op.target());
              }
            }
            break;

          case op_t::T:
          case op_t::U1:
            if (reg->bit_is_set(i, op.target())) {
              reg->amplitude[i] *= op.val();
            }
            break;

          case op_t::CU1:
            if (reg->bit_is_set(i, op.control())) {
              if (reg->bit_is_set(i, op.target())) {
                reg->amplitude[i] *= op.val();
              }
            }
            break;
        }
      }
    }
  }

  void Flush(qureg *reg) {
    Execute(reg);
    combined_gates_ += (op_list_.size() - 1);
    op_list_.clear();
  }

 private:
  int combined_gates_;
  std::vector<Op> op_list_;
};

namespace {
Jit *jit_ptr = nullptr;

Jit *jit() {
  if (!jit_ptr) jit_ptr = new Jit();
  return jit_ptr;
}
}  // namespace

void x(int target, qureg *reg) { jit()->AddOp(Op(X, target)); }

void y(int target, qureg *reg) { jit()->AddOp(Op(Y, target)); }

void z(int target, qureg *reg) { jit()->AddOp(Op(Z, target)); }

void v(int target, qureg *reg) {
  jit()->Flush(reg);
  static cmplx mv[4] = {cmplx(0.5, 0.5), cmplx(0.5, 0.5),
                        cmplx(0.5, -0.5), cmplx(0.5, 0.5)};
  for (int i = 0; i < reg->size; ++i) {
    libq_gate1(target, mv, reg);
  }
}

void yroot(int target, qureg *reg) {
  jit()->Flush(reg);
  static cmplx mv[4] = {cmplx(0.5, 0.5), cmplx(-0.5, -0.5),
                        cmplx(0.5, 0.5), cmplx(0.5, 0.5)};
  for (int i = 0; i < reg->size; ++i) {
    libq_gate1(target, mv, reg);
  }
}

void h(int target, qureg *reg) {
  jit()->Flush(reg);
  static cmplx mh[4] = {sqrt(1.0 / 2), sqrt(1.0 / 2), sqrt(1.0 / 2),
                        -sqrt(1.0 / 2)};
  libq_gate1(target, mh, reg);
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
  jit()->AddOp(Op(U1, target, cexp(gamma)));
}

void t(int target, qureg *reg) {
  jit()->AddOp(Op(U1, target, cexp(M_PI/4.0)));
}

void cu1(int control, int target, float gamma, qureg *reg) {
  jit()->AddOp(Op(CU1, control, target, cexp(gamma)));
}

void cv(int control, int target, qureg *reg) {
  jit()->Flush(reg);
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
  jit()->Flush(reg);
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
  jit()->AddOp(Op(CX, control, target));
}

void cz(int control, int target, qureg *reg) {
  jit()->AddOp(Op(CZ, control, target));
}

void ccx(int control0, int control1, int target, qureg *reg) {
  jit()->AddOp(Op(CX, control0, control1, target));
}

void flush(qureg *reg) {
  jit()->Flush(reg);  // do nothing for the non-jitted implementation.
  print_qureg_stats(reg);
  printf("# of combined gates: %d\n", jit()->combined_gates());
  delete jit();
}

}  // namespace libq
