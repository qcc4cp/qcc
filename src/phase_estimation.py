# python3
"""Example: Phase Estimation, following Nielsen/Chuang Section 5.2."""

# The terms "phase estimation" and "eigenvalue estimation" are
# often used interchangably.

# For plain phase estimation, we just apply the inverse QFT on
# a state and extract the phase. For this to work, the state must
# be in the specific shape and form that looks like what results
# from eigenvalue estimation.
#
# For eigenvalue estimation, the state is prepared with cascading
# Controlled-U's, as shown in the circuit schematic below. The
# bottom qubits are initialized with an eigenvector of U and
# with this one can then measure the corresponding eigenvalue.
#
# It appears this is the real use case, hence the terms are
# used interchangably.


from absl import app

import numpy as np
import scipy.stats

from src.lib import helper
from src.lib import ops
from src.lib import state


def expo_u(psi: state.State, u: ops.Operator, t: int) -> state.State:
  """Exponentiate U and control it from the t register."""

  # Unpack the binary fractions of the phase into the first t qubits.
  #
  # 't' qubits
  # |0> - H -------------------- ... ----o --
  # |0> - H -----------------o-- ... --- | --
  # |0> - H ---------o-------|-- ... --- | --
  # |0> - H -o-------|-------|-- ... --- | --
  #          |       |       |           |
  # 'nbits' qubits   |       |           |
  # |u> --- U^1 --- U^2 --- U^4 ... --- U^s^(t-1)
  #
  psi = ops.Hadamard(t)(psi)
  u2 = u
  for inv in reversed(range(t)):
    psi = ops.ControlledU(inv, t, u2)(psi, inv)
    u2 = u2(u2)
  return psi


def run_experiment(nbits: int, t: int = 8):
  """Run single phase estimation experiment."""

  # Make a unitary and find eigenvalue/vector to estimate.
  # We use functions from scipy for this purpose.
  #
  umat = scipy.stats.unitary_group.rvs(2**nbits)
  eigvals, eigvecs = np.linalg.eig(umat)
  u = ops.Operator(umat)

  # Eigenvalues will be of the form e^(2 i pi phi) and we want to
  # determine that value of the factor 'phi'.
  #
  # Pick single eigenvalue at 'eigen_index'
  # (any eigenvalue / eigenvector pair will work).
  #
  eigen_index = 1
  phi = np.real(np.log(eigvals[eigen_index]) / (2j * np.pi))
  if phi < 0:
    phi += 1

  # ------------------------------------------
  # Make state and circuit to estimate phi.
  # ------------------------------------------

  # Pick eigenvector 'eigen_index' to match the eigenvalue.
  # Combine the 't' register with a register wide enough to hold
  # the unitary and construct contolled gates. Also apply
  # inverse QFT.
  #
  psi = state.zeros(t) * state.State(eigvecs[:, eigen_index])
  psi = expo_u(psi, u, t)
  psi = ops.Qft(t).adjoint()(psi)

  # Find state with highest measurement probability and show results.
  #
  maxbits, maxprob = psi.maxprob()
  phi_estimate = sum(maxbits[i] * 2 ** (-i - 1) for i in range(t))

  delta = abs(phi - phi_estimate)
  print('Phase   : {:.4f}'.format(phi))
  print('Estimate: {:.4f} delta: {:.4f} probability: {:5.2f}%'
        .format(phi_estimate, delta, maxprob * 100.0))
  if delta > 0.02 and phi_estimate < 0.98:
    print('*** Warning: Delta is large')


def run_experiment_multi(nbits: int, t: int = 8):
  """Run single phase estimation experiment."""

  # This code is very similar to the above version
  # (which should be studied first).
  #
  # Above we initialized the eigenstate |u> with a
  # single eigenvector.
  #
  # Here we initialize the eigenstate |u> as a state
  # in superposition of multiple, equally weighted eigenvectors.
  #
  # This means that phase estimation should find multiple
  # states with high probability, each serving as an
  # estimate for one of the corresponding eigenvalues
  # (as binary fractions). The code in this routine is hence
  # a generalization of the above routine. Note that probability
  # is at play, sometimes this algorithm find 2 values that closely
  # match an eigenvalue, which appars to mismatch the phi's and
  # their eigenvalues. Visual inspection should be enough.

  # Make a unitary and find eigenvalue/vector to estimate.
  # This is identical to above.
  #
  umat = scipy.stats.unitary_group.rvs(2**nbits)
  eigvals, eigvecs = np.linalg.eig(umat)
  u = ops.Operator(umat)

  # Collect all eigenvalues, not just the one at 'eigen_index'
  # as above.
  #
  phi = []
  for v in eigvals:
    val = np.real(np.log(v) / (2j*np.pi))
    if val < 0:
      val += 1
    phi.append(val)
  phi = sorted(phi, key=float)

  # Superposition will be equally weighted (other probability
  # distributions are possible, but not essential for this
  # implementation).
  #
  fac = np.sqrt(1 / 2**nbits)

  # Create a state as a superposition of all the
  # eigenvectors, equally weighted by 'fac'.
  #
  ini = np.zeros(2**nbits, dtype=np.complex128)
  for idx in range(2**nbits):
    ini += fac * eigvecs[:, idx]

  # Make state and circuit to estimate phi (similar to above).
  #
  psi = state.zeros(t) * state.State(ini)
  psi = expo_u(psi, u, t)
  psi = ops.Qft(t).adjoint()(psi)

  # Find states with highest measurement probabilities and show results.
  # This should match in 'most' cases. A more sophisticated analysis to
  # match estimates and phi's is possible, but not essential for our
  # purposes here.
  #
  estimates = []
  for bits in helper.bitprod(psi.nbits):
    # The value of 0.05 is a good guestimate for the given
    # values of 't' and 'nbits' (derived experimentally).
    if psi.prob(*bits) < 0.05:
      continue

    phi_estimate = sum(bits[i] * 2**(-i-1) for i in range(t))
    estimates.append(phi_estimate)

  # Sort and make unique the values in 'estimates'.
  #
  estimates = sorted(list(set(estimates)), key=float)

  # Finally, print the phi's and estimates. They will not
  # always match perfectly.
  #
  for p in phi:
    print(f'Phase : {p:.4f} ', end='')
  print('')
  for est in estimates:
    print(f'Estim : {est:.4f} ', end='')
  marker = 'Ok' if len(phi) == len(estimates) else ''
  print(marker)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  nbits = 2
  t = 7
  print(f'Estimating {nbits} qubits random unitary eigenvalue(s) ' +
        f'with {t} bits of accuracy.')
  for _ in range(5):
    run_experiment(nbits, t)
  for _ in range(5):
    run_experiment_multi(nbits, t)


if __name__ == '__main__':
  app.run(main)
