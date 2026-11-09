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
| 48 | `assert isinstance(arg, state.State), '...')` | Stray ")" |
| 49, 52 | `return arg @ self` vs prose "Y(X) means (Y @ X)(ψ)" | Code applies Y first; code or prose must change |
| 51 | "O(2^{2.3752477})" | O(n^{2.375477}) |
| 55 * | "Y = \|+y⟩⟨−y\| − \|−y⟩⟨+y\|" | Should be \|+y⟩⟨+y\| − \|−y⟩⟨−y\| (and "Paul matrices" on the same page) |
| 57 | `Rotation(vparm, theta)` called with three arguments | Add the name parameter |

## Chapters 3-4: simulation, density matrices, decompositions

| PDF p. | Quote | Fix |
| --- | --- | --- |
| 84, 86 | "Bennet" (4x) | Bennett |
| 86 | "obtain \|f(0)⟩\|x⟩\|0⟩" | \|f(x)⟩ |
| 87 | "collapses to the measured state (postulate 5)" | Postulate 4; Born rule names the probability rule, not collapse |
| 113 | `applyc` uses undefined `qbit` | `index` |
| 123 | `assert np.allclose(...) 'Not orthonormal'` | Missing comma |
| 127 | `assert index <= 0 < nbits` | `0 <= index < nbits` |
| 128-129 | "the square of the trace ... is 0.5" | Trace of the square (purity) |
| 130 * | separable iff "the number of distinct coefficients is 1" | Nonzero coefficients; the Bell state is a counterexample as written |
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
| 238 | k "proportional to the probability pgood" | k ∝ 1/√pgood, as the next formula says |
| 241 | `u` used before defined; `nbits` undefined in `run_experiment` | Reorder; `nbits_grover` |
| 242 | "For N = 64" with four Grover qubits | N = 16 |
| 248, 250 | `find_solutions` (defined as `find_negative_solutions`); `assert solution[0]) ==` | Name; unbalanced ")" |
| 256 * | "Σ F(i)/2^{n/2} = µ" and the 2^{n/2} explanation | Amplitude is Σ F(i)/2^n, which is what the code asserts; Fig. 10.20 also shows H on the ancilla that the code never applies |
| 258 | duplicated "Note that we already know the minimum classically..." | Delete one |
| 265 | "U, U², ..., U^{t−1}" | U^{2^0}, ..., U^{2^{t−1}} |
| 273 | "highest probability (≥ 1%)" | Code uses 3% |
| 276 | "ω^n = 2^{2πin/N}" | e^{2πin/N} |
| 278 | last term "φ_{n−1}/2^{n−1}" | /2^n |
| 279 | "e^{2φi0.φ0}" (twice) | 2πi |
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
| 337-339 | edge weight multiplied into both Z factors | Hamiltonian uses w² per edge while prose and classical check use w; likely the real cause of the reported 20-30% failures |
| 342 | `-> -> List[float]` | One arrow |
| 346-348 | `z` defined, `Z` used | NameError as printed |
| 351 | "for pure states, the trace must be 1" | Trace is 1 for all density matrices; purity is the pure-state test |
| 354 * | "x − 3y + 4z = 4" with solution (1, 2, 3) | Gives 7; change the right-hand side |
| 358 | "eig returns the eigenvalues as columns"; `np.outer(v, v.adjoint())` ignores i; eigenvectors are not actually decolumned | Eigenvectors; `v[:, i]`; footnotes 9 and 11 reason from a false premise |
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
