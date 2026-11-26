# Linear Marginalization & Causal POD (`ticktack.linear`)

Shared analytical Gaussian marginalization primitives (`LinearMarginal`, `posterior`), Solin–Särkkä Hilbert-space GP projection (`solin_sarkka_basis`, `project_causal_gp`), and 24-step Smith-doubling stationary pre-history covariance (`stationary_cbm_gp_cov`). State-space Kalman GP marginalization and smoothing (`KalmanGPResult`, `kalman_gp_marginal`), block-arrow solvers, and canonical POD modes live in [`ticktack.kalman`](#state-space-kalman-gp-block-arrow-solvers-and-canonical-pod-ticktackkalman); their old `ticktack.linear` names remain as temporary aliases. See [Gaussian Processes & Information Field Theory](../gp_and_ift.md).

---

::: ticktack.linear
    options:
      show_root_heading: false

---

## State-space Kalman GP, block-arrow solvers, and canonical POD (`ticktack.kalman`)

Continuous-time Joseph-form Kalman filter and Rauch–Tung–Striebel / Bryson–Frazier smoother (`KalmanGPResult`, `kalman_gp_marginal`), block-arrow linear-Gaussian solvers (`ArrowProblem`, `arrow_problem`, `arrow_at`), and canonical data- and prior-orthogonal Wiener SVD modes (`CanonicalPODResult`, `canonical_pod_modes`).

::: ticktack.kalman
    options:
      show_root_heading: false
