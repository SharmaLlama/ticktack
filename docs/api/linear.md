# Linear Marginalization & Causal POD (`ticktack.linear`)

Shared analytical Gaussian marginalization primitives (`LinearMarginal`, `posterior`), Solin–Särkkä Hilbert-space GP projection (`solin_sarkka_basis`, `project_causal_gp`), 24-step Smith-doubling stationary pre-history covariance (`stationary_cbm_gp_cov`), and Joseph-form state-space Kalman marginalization (`kalman_gp_marginal`). The block-arrow solvers and canonical POD modes now live in [`ticktack.kalman`](#block-arrow-solvers-and-canonical-pod-ticktackkalman); their old `ticktack.linear` names remain as temporary aliases. See [Gaussian Processes & Information Field Theory](../gp_and_ift.md).

---

::: ticktack.linear
    options:
      show_root_heading: false

---

## Block-arrow solvers and canonical POD (`ticktack.kalman`)

Block-arrow linear-Gaussian solvers (`ArrowProblem`, `arrow_problem`, `arrow_at`) and canonical data- and prior-orthogonal Wiener SVD modes (`CanonicalPODResult`, `canonical_pod_modes`).

::: ticktack.kalman
    options:
      show_root_heading: false
