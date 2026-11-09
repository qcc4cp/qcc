# Quantum Computing for Programmers, 2nd Edition: Errata

This review was created by Claude Fable 5.1 Medium.
Page numbers are PDF pages (book page = PDF page − 18).
Note that the printed book's numbers may differ slightly.

I (Robert Hundt) didn't use AI while writing the book itself.
All mistakes and errors are mine and I apologize to potentially misleading
readers.

## Chapters 1-2: foundations

| PDF p. | Quote | Fix |
| --- | --- | --- |
| 20 | prose "x = 1 + 2i", code "1.0 + 0.5j" | Make prose and code agree |
| 23 | "\|y⟩ is a 1 x 3 vector, and ⟨y\| is a 3 x 1 vector" | Dimensions swapped |
| 25 | positive semidefinite "if all its eigenvalues are positive" | Non-negative |
| 32 | `((x == 1) or (x == 0)).all()` | `or` on numpy arrays raises; use `\|` |
| 37 * | "e^{π/2}", "e^{3π/2}" in the y-basis equations | Missing i: e^{iπ/2}, e^{i3π/2} |
| 38 | "tensor product in Section 1.3" | Section 1.5 |
| 40-47 | `cls.name = op_name` in `Tensor.__new__` | Assigns a class attribute, so every tensor without a name of its own reports the name of the most recently created tensor (e.g. after creating "foo" and then "bar", the first reports "bar"). Assign `obj.name` on the new instance, and keep a class-level default `name = None` for arrays numpy derives from a tensor. |
| 48 | `assert isinstance(arg, state.State), '...')` | Stray ")" |
| 49, 52 | `return arg @ self` vs prose "Y(X) means (Y @ X)(ψ)" | Code applies Y first; code or prose must change |
| 51 | "O(2^{2.3752477})" | O(n^{2.375477}) |
| 55 * | "Y = \|+y⟩⟨−y\| − \|−y⟩⟨+y\|" | Should be \|+y⟩⟨+y\| − \|−y⟩⟨−y\| (and "Paul matrices" on the same page) |
| 57 | `Rotation(vparm, theta)` called with three arguments | Add the name parameter |
| 58-70 * | `Toffoli(idx0, idx1, idx2)` built as `ControlledU(idx0, idx1, Cnot(idx1, idx2))` | Only correct when the three qubits are ordered (idx0 < idx1 < idx2 or idx0 > idx1 > idx2): the inner `Cnot` is assumed to start at idx1 and to lie entirely on one side of the outer control. For the other four orders of three adjacent qubits, e.g. (0, 2, 1), the result has the wrong width (16 x 16 instead of 8 x 8); of the 120 ordered triples of distinct indices below 6, only 40 are correct. Keep the construction, but choose the indices so that the outer control is at one end of the three qubits: use as outer control the one farther from the target (with the target in the middle either works), and pass the qubit of the inner `Cnot` that is nearer to it as second argument of `ControlledU`, because `ControlledU(c, t, u)` takes the qubit of u nearest to the control. For ordered qubits this is the printed call. |

## Chapters 3-4: simulation, density matrices, decompositions

| PDF p. | Quote | Fix |
| --- | --- | --- |
| 84, 86 | "Bennet" (4x) | Bennett |
| 86 | "obtain \|f(0)⟩\|x⟩\|0⟩" | \|f(x)⟩ |
| 87 | "collapses to the measured state (postulate 5)" | Postulate 4; Born rule names the probability rule, not collapse |
| 100-122 * | `isinstance(idx_set, int)` in `apply1` and `isinstance(ctl, int)` in `_ctl_by_0` | A numpy integer is not an `int`. `c.x(np.int64(1))` silently does nothing (as does a tuple or array of indices), `cx(np.int64(0), 1)` raises an `IndexError`, and a control-by-0 such as `cx([np.int64(0)], 1)` silently fires without flipping the control, which gives a wrong state. Test for `numbers.Integral`, convert the indices to `int`, and raise a `TypeError` for unsupported index types instead of ignoring them. |
| 113 | `applyc` uses undefined `qbit` | `index` |
| 123 | `assert np.allclose(...) 'Not orthonormal'` | Missing comma |
| 127 | `assert index <= 0 < nbits` | `0 <= index < nbits` |
| 127-128 * | `TraceOut` decrements every later index (`index_set[i] = index_set[i] - 1`) after tracing out a qubit | Only correct for ascending indices. For descending or unordered indices it either fails an assertion (tracing out [2, 0] makes the second index −1) or silently returns the wrong reduced density matrix (e.g. [3, 1]); it also overwrites the caller's list ([0, 1, 2] becomes [0, 0, 0]) and fails for tuples. Shift an index only by the number of already traced out qubits below it, and work on a copy of the list. |
| 128-129 | "the square of the trace ... is 0.5" | Trace of the square (purity) |
| 130 * | separable iff "the number of distinct coefficients is 1" | Nonzero coefficients; the Bell state is a counterexample as written |
| ~130 * | Schmidt decomposition: `u0, d0, _ = np.linalg.svd(rho0)`, `a1, _, _ = np.linalg.svd(rho1)`, and `np.kron(u0[:, i], a1[i, :])` | The two bases come from separate SVDs of the reduced density matrices. Each basis vector is only determined up to a phase (a sign for real states), so u_i and v_i do not match, and `a1[i, :]` is a row, while SVD returns the vectors as columns. The reconstructed state is wrong for 145 of 200 random real states and for all 200 random complex ones (also for separable complex states). The examples pass only because their amplitudes happen to give matching signs. Take both bases from a single SVD of the 2x2 matrix of amplitudes, psi = U D V^†: u_i are the columns of U, v_i the rows of V^†, and the Schmidt coefficients are D (with D² equal to the eigenvalues), i.e. `newpsi = d0[0] * np.kron(u0[:, 0], v0[0, :]) + d0[1] * np.kron(u0[:, 1], v0[1, :])`. The check tolerance can then be 1e-5 instead of 1e-3. |
| ??? | Purification: `np.linalg.eig(rho)`, `np.trace(mat@mat)` and `np.dot(mat, mat.transpose())` for the reduced density matrix | Fails for complex density matrices; the examples only use real ones. The checks need the conjugate transpose: the norm of the purified state is `trace(mat @ mat.conj().T)`, and the reduced density matrix is `mat @ mat.conj().T`. Use `np.linalg.eigh`, which returns orthonormal eigenvectors for a Hermitian matrix: `eig` does not for repeated eigenvalues (such as the zero eigenvalues of a pure state), and in single precision the purification is then wrong. Clamp the eigenvalues at 0, and use a tolerance of about 1e-6 in the comparisons. With these changes, 1800 of 1800 random complex pure, mixed and degenerate states pass, against none of 10 complex test cases before (the real ones passed). The comment "rho = sum_k sqrt(pk) \|psi_k⟩ tensor \|psi_k⟩" should read \|psi⟩ = ... |
| 131-138 | `print(f'{i:11.2f} I + {x:11.2f} X + ...')` in the Pauli representation of a single qubit | `i` is the loop counter, so the identity coefficient prints as 0.00, 1.00, 2.00, ... The coefficient is `c`, which is always 1. |
| 139 | "det U = a^{i2α}"; `if a >= b` on complex a, b | e^{i2α}; `abs(a) >= abs(b)` |
| 142 | "Cout will also start as \|1⟩" | Starts as \|0⟩ |
| 146 | integer factorization given as NP-hard | Not known to be NP-hard |
| 147 | "A few days versus 10,000 years is another factor of 1,000" | About 10^6; the paragraph's arithmetic is off by 10^3 |

## Chapters 5-9: entanglement, Deutsch-Jozsa, state preparation, Solovay-Kitaev

| PDF p. | Quote | Fix |
| --- | --- | --- |
| 164 | "CHSH equality" | Inequality |
| 165 * | "\|β00⟩ = \|Ψ+⟩ = (\|00⟩+\|11⟩)/√2" | \|Φ+⟩ (as on pp. 156, 162); also "computation basis" |
| 166 | x=0,y=1 case: Bob measures "\|b1⟩" with cos²π/8 | \|b0⟩; and for x=1 Alice measures \|+⟩, not \|1⟩ |
| 184 | `def check_result(...) -> None` | Missing colon |
| 187 | "f:{0,1}^N → {0,1}, N = 2^n" and "Deutsch-Jozska" | {0,1}^n; Jozsa |
| 187-190 | `make_c` in the Bernstein-Vazirani listing loops `for idx in range(nbits - 1)` | The argument is the length of the secret (callers pass the qubit count minus 1), so the last bit of the secret is never set: it is always 0, and only half of all secrets can occur (64 of 128 for 7 bits). Loop over `range(nbits)`. The oracles themselves are correct for every secret. |
| 188-195 * | Simon oracle built for a secret c "with the msb set to 1", but `make_c` samples every bit independently; comment "return x % c" | For a nonzero c with msb 0 (1 of 4 secrets for 2 bits, 3 of 8 for 3 bits) the oracle is one-to-one, f(x) ≠ f(x ⊕ c), and half of the measurements violate ⟨z, c⟩ = 0 (mod 2). Set the msb of c to 1 in `make_c` and reject other nonzero secrets in `make_u`. The oracle returns x XOR c, not x % c. Reported by [REPORTER NAME]. |
| 196 | "0.082 ≃ \|00001⟩ ... (∆ = 0.0195)" | Vector entry is 0.1, ∆ = 0.0375 |
| 198 | "for a time-dependent Hamiltonian, U(t) = e^{−itH}" | Time-independent; the proof's sign also disagrees with the claim |
| 199 | Figure 9.1 last gate labeled w2 | w3 |
| 214 | `0.5 - 0.5 * np.cos(theta) / 2` | `0.5 - 0.5 * np.cos(theta / 2)` per the derived formula |

## Chapters 10-11: Grover, amplitude estimation, QPE, QFT, Shor

| PDF p. | Quote | Fix |
| --- | --- | --- |
| 232 | `make_f` uses undefined `nsolutions` | Rename the parameter |
| 232-235 * | `grover = inversion(uf)` | Composition reverses (`x(y)` is `y @ x`, p. 49), so this applies the inversion before the oracle. The first inversion leaves the uniform state unchanged, so k iterations act like k − 1: the marked state gets sin²((2k−1)θ) instead of sin²((2k+1)θ), e.g. 0.25 instead of 1.0 for N = 4 (the assertion then fails for 3 of 4 targets). Use `uf(inversion)`. The same line in `sat3.py`, `minimum_finding.py` and `state_prep.py`; `counting.py` and `amplitude_estimation.py` are not affected (phase estimation only needs the eigenvalues, which both orders share). Reported by github user WindOctober. |
| 235 | "we apply Z gates to the qubit indices that represent a 0" | Code applies X gates |
| ??? * | Graph coloring: `qc.multi_control(chk, res, tmp, ops.PauliZ(), 'multi')` with the result qubit `res` in \|0⟩, `iterations = int(math.pi / 4 * math.sqrt(2**(g.num*2)))`, and a check that only the most likely states are valid colorings | This is not a Grover search. A controlled Z on a qubit in \|0⟩ does nothing, so the oracle never marks the solutions, and the probability of the solutions changes erratically with the number of iterations (0.37 to 0.53 for the examples, instead of 0.96 to 1.0). Put `res` in \|−⟩ (X, then H) and use a controlled X, which then flips the phase of the solutions. The number of iterations must use the number of solutions M, π/4 √(N/M); the printed formula assumes M = 1, but the examples have 4 solutions, and it overshoots. The check passes because the most likely states happen to be valid colorings even without amplification; also check that all solutions are found and that their total probability is high. The docstring of `Graph.verify` has the logic inverted (it returns True if some connected vertices have different colors), and the test of the equality circuit only checks the result qubit for equal pairs. |
| 238 | k "proportional to the probability pgood" | k ∝ 1/√pgood, as the next formula says |
| 241 | `u` used before defined; `nbits` undefined in `run_experiment` | Reorder; `nbits_grover` |
| 242 | "For N = 64" with four Grover qubits | N = 16 |
| 242-243 * | `phi_estimate = helper.bits2frac(maxbits)`, with `maxbits` from `psi.maxprob()`, and the printed `prob` taken from `maxprob` | `maxbits` has the bits of all qubits (phase register, oracle register and ancilla), so the other qubits are read as additional binary digits of the phase. This shifts the estimate by up to one bin, 2^−nbits_phase, and the count is wrong when the shift exceeds the rounding margin: with 6 phase qubits and M = 10, 3 of 4 seeds fail. With 7 phase qubits the program passes, but uses up to 0.43 of the 0.5 allowed for rounding, against 0.17 with the fix. Use only the first `nbits_phase` bits. The printed probability is that of a single basis state of all qubits and not that of the phase estimate; sum it over the other qubits (the result is at most about 50%, as the probability is shared with the conjugate phase 1 − φ). |
| 243-247 * | Amplitude estimation prepares `psi` with `ops.Hadamard(nbits_phase + nbits_grover + 1)(psi)`, i.e. Hadamard gates on all qubits | The data register must start in A\|0⟩, not in the equal superposition: the two eigenvalues of the Grover operator that AE reads out belong to the plane spanned by the good and bad parts of A\|0⟩. The two states coincide only for A = H^⊗n, which is what the listing uses. For other A the result is wrong; e.g. for a rotated A with true amplitude 0.478, the program returns 1.0 (17 or 18 of 18 runs failed for each of several kinds of A). Prepare the phase register with Hadamards, the data register as A\|0⟩, and the ancilla as H\|1⟩ = \|−⟩. |
| 248, 250 | `find_solutions` (defined as `find_negative_solutions`); `assert solution[0]) ==` | Name; unbalanced ")" |
| 256 * | "Σ F(i)/2^{n/2} = µ" and the 2^{n/2} explanation | Amplitude is Σ F(i)/2^n, which is what the code asserts; Fig. 10.20 also shows H on the ancilla that the code never applies |
| 258 | duplicated "Note that we already know the minimum classically..." | Delete one |
| 259-263 * | Median via `mean = np.mean(diffnorm)` with `diffnorm = np.linalg.norm(diff)`; check `max(np.mean(x), x[median]) / min(np.mean(x), x[median]) > 1.02` | The median minimizes the sum of absolute differences, but the norm is an L2 cost, whose minimizer is the element nearest the mean of x, and the check compares against the mean. It only passes for large uniform data, where mean and median are close (for [1, 2, 3, 4, 100, 2, 3, 3] it picks 4, the median is 3); it also rejects correct medians, e.g. 173 of 200 random inputs with 2 qubits. Use the mean of the normalized difference vector times its norm (as in the mean computation), and compare the sum of absolute differences against that of `np.median(x)`; for an even number of elements both middle values are optimal. |
| 265 | "U, U², ..., U^{t−1}" | U^{2^0}, ..., U^{2^{t−1}} |
| 273 | "highest probability (≥ 1%)" | Code uses 3% |
| ~273 * | `run_experiment_multi`: `estimates = [helper.bits2frac(bits) for bits in helper.bitprod(psi.nbits) if psi.prob(*bits) >= 0.03]` | `bits` covers all qubits, the t phase qubits and the data qubits, so the data qubits are read as additional binary digits of the phase, and the probability is that of a single basis state of all qubits instead of that of the phase estimate. The data register is in a superposition of eigenvectors, so the probability of an estimate is spread over the data states and can fall below the threshold. The example (2 data qubits, t = 7) finds all eigenvalues, but only because the tolerance of 0.01 exceeds a bin; with t = 6, 7 of 200 eigenvalues are missed, and with 3 data qubits 165 of 400. Loop over `helper.bitprod(t)` and add up the probabilities over the data qubits, `sum(psi.prob(*bits, *data) for data in helper.bitprod(nbits))`; then all are found (except phases within half a bin of 1, whose estimate is 0). |
| 276 | "ω^n = 2^{2πin/N}" | e^{2πin/N} |
| 278 | last term "φ_{n−1}/2^{n−1}" | /2^n |
| 279 | "e^{2φi0.φ0}" (twice) | 2πi |
| 281-285 * | `arith_quantum_constant`: `precompute_angles(c, n)` and `maxbits[0:n]` | The register has n + 1 qubits, the top one being the carry out, but only n rotations are applied and only n result bits are read. Every sum of 2^n or more is wrong: 7 + 7 gives 6 for n = 3, and for n = 2 to 5 exactly the pairs with a sum of at least 2^n fail (28 of 64 for n = 3). Use `precompute_angles(c, n + 1)` and read n + 1 bits. The examples only add numbers whose sum is below 2^n. |
| 285-290 * | `arith_quantum_mult`: the loops in `add_src_to_targ` run over `range(nbits)`, and the result is read as `c[0] : c[0 + nbits_b * 2]` | The registers have nbits + 1 qubits (the addition loops over `range(n + 1)`), so the Fourier transform skips qubit 0 and its inverse skips the top qubit. This is not an adder if the target is not zero: 3 × 3 with 2 bit operands gives 10 (with probability 0.38) instead of 9. The top qubit of the result register is not read either, which drops the top bit of products of at least 2^(2·nbits_b) (6 × 3 gives 2). Loop over `range(nbits + 1)` and read `c[0] : c[nbits_b * 2] + 1`. The examples only multiply with a < 3, where a single addition to zero happens to work. |
| 291 | Eq. (11.11) "(x + y) mod N ≡ x mod N + y mod N" | Needs the outer mod N |
| 296 | `while 1 != (num ** order) % modulus` | Missing colon |
| 299 | two different vectors both called \|u1⟩ | Uniform superposition is \|u0⟩ |

## Chapters 12-16: QAOA, QML, noise, languages

| PDF p. | Quote | Fix |
| --- | --- | --- |
| 315 | `idx_bits0` undefined | `idx_bits` |
| 322 | Hamiltonian built from PauliY, label prints "X" | Make consistent |
| 326 * | "S†HZHS" for the Y measurement | Equals −Y; correct is S H Z H S† (circuit, code and Table 13.1 are already right) |
| 336 | "decreases the energy for the vertices in the same set"; "mirroring Equation (13.2)" | Opposite; Eq. (13.3) |
| 336-337 | `random.sample(range(0, 3 + i - 1), 2)` in the random graph builder | `range` excludes its end, so the newest existing node 2 + i is never chosen as the target of an edge (with 8 nodes, node 6 never gets an edge to node 7). Use `range(0, 3 + i)`. |
| 337-339 * | edge weight multiplied into both Z factors | Hamiltonian uses w² per edge while prose and classical check use w. Confirmed as the cause of the reported failures: on the same graphs, a Hamiltonian with weight w finds the maximum cut in every run, while the printed one fails in 11-19% of the runs with 6 to 10 nodes. Apply the weight once per edge, e.g. build the diagonal of Z⊗Z and scale it by w. |
| 337-339 * | `graph_to_hamiltonian`: `Identity(idx2 - idx1 + 1)` and `Identity(n - idx2 + 1)`, with idx1 the larger and idx2 the smaller node index | The numbers of identity gates between and after the two Z gates are wrong (zero or negative, and off by one), the matrices do not fit, and the function raises an exception. Use `Identity(lo)`, Z, `Identity(hi - lo - 1)`, Z, `Identity(n - hi - 1)` for the nodes lo < hi, and multiply the weight in once. |
| 341-343 * | `random.sample(range(1, nmax), nnum)` and `tensor_diag` with `diag = [1, 1]` and `for i in range(1, n)` | The number nmax has no qubit and is silently ignored: with nmax = 15, the set {5, 10, 15} is reported to have no partition, although 5 + 10 = 15. Give the number k its own qubit k − 1 (start from `diag = [1]`, loop to n), sample from `range(1, nmax + 1)`, and reject numbers outside 1..n. |
| 342 | `-> -> List[float]` | One arrow |
| 346-348 | `z` defined, `Z` used | NameError as printed |
| 349-353 * | `eucl_dist_q = (4 * z * (p0 - 0.5)) ** 0.5` and `assert np.allclose(eucl_dist_q, eucl_dist_c, atol=1e-4)` | For equal or nearby vectors the exact radicand is 0 or tiny, and rounding makes it slightly negative about half of the time. The root is then NaN and the assertion fails. The precision of p0 limits the error of the squared distance (about 10^−6 · z), not of the distance, so a fixed tolerance on the distance cannot hold for close vectors. Clamp the radicand at 0 and compare the squared distances with a tolerance proportional to z. |
| 351 | "for pure states, the trace must be 1" | Trace is 1 for all density matrices; purity is the pure-state test |
| 354 * | "x − 3y + 4z = 4" with solution (1, 2, 3) | Gives 7; change the right-hand side |
| 355-357 * | `gamma = np.array([np.dot(psi.conj(), v[i]) for i in range(dim)])`, and `v[i]` in the reconstruction of psi and in `apply_hamiltonian` | `np.linalg.eig` returns the eigenvectors as the columns of `v`: the eigenvector of `lam[i]` is `v[:, i]`, while `v[i]` is a row. The check at t = 0 cannot see this, because the rows of a unitary matrix are also an orthonormal basis and psi is still reconstructed. For t > 0 the phases are attached to the wrong vectors: the printed evolution differs from `expm(-iAt)psi` by up to 1.5 for random Hermitian matrices, and is only correct by accident for real symmetric matrices such as the Morrell/Zaman/Wong example. Use `v[:, i]` in all three places. |
| 358 | "eig returns the eigenvalues as columns"; `np.outer(v, v.adjoint())` ignores i; eigenvectors are not actually decolumned | Eigenvectors; `v[:, i]`; footnotes 9 and 11 reason from a false premise |
| 359-362 * | `res = (qc.psi > 0.001).nonzero()[0]` | numpy orders complex numbers by their real part first, so amplitudes with a negative real part are dropped. A solution with a negative component then crashes with an IndexError. Compare the magnitudes: `np.abs(qc.psi) > 0.001`. |
| 363 | "λ_i = e^{2πφ_i}" | Missing i; λ_i also reused for two different things |
| 369 | Table 15.1 nuclear spin τ_Q "10^−2 to 10^−8"; "almost ten orders of magnitude" | 10^8; about fifteen |
| 372 | "C(ρ) = (1 − p)ρ + pXρX" with E_0 = √p I | Swap p and 1 − p in one of the two |
| 374 | "normally distributed random number" from `random.random()` | Uniform |
| 376, 422 | "Rolf, 1995" / "Rolf, L." | Landauer, R. |
| 383-384 | "micro-Kelvin temperatures" | Milli-Kelvin |
| 388 | "CQTC" | CTQC |
| 395 | eigenvalue e^{iφ} "(which means U\|ψ⟩ = \|ψ⟩)" | U\|ψ⟩ = e^{iφ}\|ψ⟩ |
| 399 | "IBMQX5 has five qubits" | The cited device is the 16-qubit Rueschlikon |
| 411 | `Break;` | `break;` |
