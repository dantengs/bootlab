# Regression: ising-oe, Λ = 11, the first three Skydive steps

`bt-steps.json`: the first three steps of `examples/ising` (ising-oe, Λ = 11, simpleboot blocks, start
(0.519, 1.4, 0.956), the reference program's Skydive settings: gap upper limit 0.1 at every step, ε = 1e-20,
Hessian guess (1e6, 1e4, 1e3)) as the reference Hyperion-based program (bt r3.8) computed them, read from its
worker log at full precision. Per step: the point x, terminateReason, N, primal, dual, gap, Δx, BFGS gradient
and Hessian, the total iteration count, α, β, climbedQ, findMinimumQ. The reference shows x, Δx and the
gradient as 512-bit floating point (about 154 digits).

`bin/bootlab regress` runs `examples/ising` for three steps on one node and compares each step with this file:
every number to 130 digits or more, the reasons, iteration counts, climbedQ and findMinimumQ equal
(docs/en/interfaces.md section 16). From step 2 on the centre differs from the reference's by the last digit of
Δx (bootlab adds exactly in decimal, the reference in 512-bit floating point), so N agrees to about 140 digits
and Δx, gradient and Hessian to about 134.
