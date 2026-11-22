# Solvers & Carbon Box Models

`ticktack 2.0` separates the **physical reservoir topology and rate parametrization** (`ticktack.cbm`) from the **time-propagation operator** (`ticktack.solvers`). Because pre-industrial and Holocene carbon box models are linear time-invariant (LTI) systems with timescales spanning four orders of magnitude – from $\sim 0.2\text{ yr}$ stratosphere–troposphere exchange to $\sim 5000\text{ yr}$ abyssal ocean turnover and $8267\text{ yr}$ radioactive mean life – exploiting exact matrix exponentials and frequency-domain resolvents yields orders-of-magnitude gains in both accuracy and speed over generic Runge–Kutta stepping.

---

## 1. Perron–Frobenius Carbon Box Models (`ticktack.cbm`)

### Graph Topology & Structural Validation

Every `CarbonBoxModel` holds a static `Topology` PyTree defining the directed reservoir graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, tropospheric box indices $\mathcal{T}$, hemispheric routing tags (`"north"` / `"south"`), and normalized cosmic-ray production allocation $\mathbf{b}$ ($\sum_i b_i = 1$). On construction, `Topology` runs [Tarjan's strongly connected components](https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm) (SCC) algorithm to verify graph reachability from every production-receiving reservoir ($b_i > 0$) and ensure the generator $A$ is nonsingular.

### Constrained Rate Parametrizations

Rather than exposing raw matrix entries that can violate mass conservation or positivity during gradient steps, `ticktack 2.0` provides six immutable `CarbonBoxModel` classes:

| Class | Free Parameters | Physical Constraint Enforced | Typical Use Case |
| :--- | :--- | :--- | :--- |
| `FixedCBM` | $0$ | Frozen reservoir masses $\mathbf{M}$ and fluxes $F_{i \to j}$ | Standard Miyake / solar / IFT production inference with fixed `Brehm21` or `Guttler15` transport |
| `BalancedCBM` | $|\mathcal{E}|$ directed log rates (`log_rates_free`, or $p \le |\mathcal{E}|$ tied groups) + $\ln M_{\text{tot}}$ | Exact pre-industrial $^{12}\text{C}$ mass balance $\sum_j M_j k_{j \to i} = M_i \sum_j k_{i \to j}$ via [Perron–Frobenius](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem) null-vector stationary masses | Sensitivity analysis & Fisher spectrum on directed multi-box topologies (`Brehm21` has 56 directed free rates, or 52 under N/S symmetry tying) |
| `ReversibleCBM` | $(n - 1)$ log mass ratios + $|\mathcal{E}_{\text{undir}}|$ symmetric log fluxes (`log_sym_fluxes`) + $\ln M_{\text{tot}}$ | Detailed balance $M_i k_{i \to j} = M_j k_{j \to i}$, guaranteeing real negative eigenvalues of $A$ ($M^{1/2}$ similarity to a negative-definite symmetric matrix) | Non-oscillatory reversible carbon-exchange priors |
| `OpenCBM` | Directed log rates + non-negative sink vector $\mathbf{s} \ge 0$ | Irreversible sediment burial / carbonate weathering sinks on top of closed transport | Million-year / glacial–interglacial open-system experiments |
| `EffectiveCBM(K)` | $2K - 2$ unconstrained log rates | Minimal star-graph topology with zero similarity-gauge redundancy; see [Fisher Information & EffectiveCBMs](fisher_and_effective_cbm.md) | Joint carbon-cycle + production inference & Bayesian model comparison |
| `SubspaceCBM` (`FisherSubspaceCBM`) | $m$ unconstrained mode amplitudes `alpha` | Exact Perron–Frobenius mass balance restricted to the top $m$ Schur-marginalized Fisher log-rate eigenmodes (`SubspaceCBM.from_fisher`) | Joint full-topology uncertainty propagation without sampling nullspace directions |

In `BalancedCBM`, unconstrained coordinates `log_rates_free` parametrize the directed first-order transfer rates $k_{i \to j} > 0$ (optionally tied across symmetric hemispheric edges via `tie_matrix`) on any strongly connected graph $\mathcal{G}$; solving for the unique positive [Perron–Frobenius](https://en.wikipedia.org/wiki/Perron%E2%80%93Frobenius_theorem) right null-vector of the conservative rate matrix yields the stationary reservoir $^{12}\text{C}$ fractions $M_i / M_{\text{tot}}$ automatically so Kirchhoff inflow equals outflow at every node without constrained optimization.

---

## 2. Exact LTI Propagation (`ExactLinearSolver`)

`ExactLinearSolver` is the default solver on `RadiocarbonModel`. Writing $\mathbf{y}(t) = \mathbf{y}_{\text{ss}}(Q_{\text{std}}) + \delta\mathbf{y}(t)$ and $\delta Q(t) = Q(t) - Q_{\text{std}}$, linearity of $\dot{\mathbf{y}} = A\mathbf{y} + \alpha_{\text{unit}} Q(t)\mathbf{b}$ splits the perturbation $\delta\mathbf{y}(t)$ into exact additive contributions from each constituent `ProductionTerm` in a `ProductionSum`:

1. **Gridded / continuous forcing** via Van Loan first-order-hold (FOH) propagation and causal `rfft` convolution.
2. **Impulsive Miyake spikes** (`MiyakeSpike`) via analytical sub-interval clamping and Gauss–Legendre quadrature.
3. **Harmonic solar cycles** (`HarmonicSolar`) via the exact complex frequency-domain resolvent – **zero burn-in**.

### Van Loan $(n+2)\times(n+2)$ First-Order-Hold Matrix Exponential

Suppose continuous cosmic-ray forcing $\delta Q(t)$ is sampled on a uniform time step $\Delta t$ (`dt = 1/12` yr by default) and interpolated piecewise linearly across $[t_k, t_{k+1}]$:

$$
\delta Q(t_k + \tau) = \delta Q_k + \frac{\delta Q_{k+1} - \delta Q_k}{\Delta t}\,\tau, \qquad \tau \in [0, \Delta t].
$$

Following [Van Loan (1978)](https://doi.org/10.1109/TAC.1978.1101743), `ExactLinearSolver` forms a single $(n + 2) \times (n + 2)$ block upper-triangular matrix exponential

$$
\exp\!\left(
\begin{bmatrix}
A & \mathbf{b} & \mathbf{0} \\
\mathbf{0}^\top & 0 & 1/\Delta t \\
\mathbf{0}^\top & 0 & 0
\end{bmatrix} \Delta t
\right)
=
\begin{bmatrix}
\Phi(\Delta t) & \boldsymbol{\gamma}_0(\Delta t) & \boldsymbol{\gamma}_1(\Delta t) \\
\mathbf{0}^\top & 1 & 1 \\
\mathbf{0}^\top & 0 & 1
\end{bmatrix},
$$

where $\Phi(\Delta t) = e^{A\Delta t}$, $\boldsymbol{\gamma}_0(\Delta t) = \int_0^{\Delta t} e^{A(\Delta t - \tau)}\mathbf{b}\,\mathrm{d}\tau$, and $\boldsymbol{\gamma}_1(\Delta t) = \frac{1}{\Delta t}\int_0^{\Delta t} \tau\,e^{A(\Delta t - \tau)}\mathbf{b}\,\mathrm{d}\tau$. This yields the **exact** single-step recurrence

$$
\delta\mathbf{y}_{k+1} = \Phi(\Delta t)\,\delta\mathbf{y}_k + \alpha_{\text{unit}}\!\left[\big(\boldsymbol{\gamma}_0(\Delta t) - \boldsymbol{\gamma}_1(\Delta t)\big)\delta Q_k + \boldsymbol{\gamma}_1(\Delta t)\,\delta Q_{k+1}\right]
$$

without ever inverting $A$ or suffering small-$\Delta t$ catastrophic cancellation in $(e^{A\Delta t} - I)A^{-1}$.

When the CBM parameters are fixed across a long grid of $N_t$ steps, the recurrence is a causal LTI discrete convolution $\delta y_{c,k} = \sum_{m=0}^k h_{c,m}\,\delta Q_{k-m}$ with impulse kernel $h_c$, which `ExactLinearSolver` evaluates in $\mathcal{O}(N_t \log N_t)$ operations via zero-padded real fast Fourier transforms (`jnp.fft.rfft` / `irfft`) while falling back to `jax.lax.scan` on short grids.

### Exact Sub-Interval Spike Quadrature (No Grid-Snapping Ringing)

Impulsive Miyake events have durations $\tau_{\text{dur}} \sim 0.05$–$1.5\text{ yr}$ whose onset $t_{\text{start}}$ and end $t_{\text{start}} + \tau_{\text{dur}}$ move continuously during `jax.grad`-based optimization (`fit_map`, `sample_nuts`, `sample_smc`). Evaluating a discontinuous `MiyakeSpike` on a fixed discrete time grid causes zero-gradient plateaus and step-discontinuity ringing whenever $t_{\text{start}}$ crosses a knot.

Instead, `ExactLinearSolver` integrates `MiyakeSpike` analytically:

- **Boxcar profile (`profile="boxcar"`)**: For any evaluation point $t$ or seasonal growth window $[a, b]$, overlap intervals $[t_1, t_2] = [t_{\text{start}}, t_{\text{start}} + \tau_{\text{dur}}] \cap (-\infty, t]$ are computed with smooth `jnp.clip` bounds and propagated in closed form via block matrix exponentials $\int e^{A\tau}\mathbf{b}\,\mathrm{d}\tau$ and double integrals $\iint e^{A\tau}\mathbf{b}\,\mathrm{d}\tau\,\mathrm{d}s$ for window averages. Consequently, $\Delta^{14}\text{C}(t)$ is $C^1$-smooth with respect to $(t_{\text{start}}, \tau_{\text{dur}}, \text{fluence})$ everywhere!
- **Smooth profiles (`"supergaussian"`, `"skewgaussian"`, `"gamma"`)**: Integrated over their smooth dimensionless support sub-panels (`MiyakeSpike.unit_support_panels`) with fixed-order Gauss–Legendre quadrature, maintaining machine-precision gradients down to arbitrary sub-monthly FWHM `duration`.

### Complex Harmonic Solar Resolvent (Eliminating the 2000-Year Burn-In)

In `ticktack 1.x`, simulating a stationary 11-year solar cycle $\delta Q_{\text{solar}}(t) = \sum_m \big[a_m \cos(\omega_m(t - t_{\text{ref}})) + c_m \sin(\omega_m(t - t_{\text{ref}}))\big]$ required integrating the stiff ODE forward for $\sim 2000\text{ years}$ prior to the data window so deep-ocean transient modes could decay.

Because $A$ is strictly Hurwitz, the exact periodic steady-state response to $e^{i\omega_m(t - t_{\text{ref}})}$ is given immediately by the **complex resolvent vector**

$$
\mathbf{r}(\omega_m) = \alpha_{\text{unit}}\,\big(i\omega_m I_n - A\big)^{-1}\mathbf{b} \in \mathbb{C}^n.
$$

Writing $z_m(t) = e^{i\omega_m(t - t_{\text{ref}})}$, the instantaneous response at any epoch $t$ is $\operatorname{Re}\!\big[(a_m - i c_m)\,\mathbf{r}(\omega_m)\,z_m(t)\big]$, and its exact average over any tree-ring photosynthesis window $[t_a, t_b]$ of midpoint $t_{\text{mid}} = \tfrac{1}{2}(t_a + t_b)$ and width $\Delta w = t_b - t_a$ acquires the closed-form `sinc` attenuation factor

$$
\frac{1}{\Delta w}\int_{t_a}^{t_b} e^{i\omega_m(t - t_{\text{ref}})}\,\mathrm{d}t
= e^{i\omega_m(t_{\text{mid}} - t_{\text{ref}})}\,\operatorname{sinc}\!\left(\frac{\omega_m \Delta w}{2\pi}\right),
$$

where $\operatorname{sinc}(x) = \sin(\pi x)/(\pi x)$ (`jnp.sinc`). Evaluating `HarmonicSolar` over $S$ series and $N_{\text{obs}}$ rings requires one $n \times n$ complex linear solve per frequency – reducing cost from thousands of ODE years to microseconds with zero transient error.

---

## 3. Non-Autonomous ODE Solvers (`DiffraxSolver` & `ModaxSolver`)

When reservoir $^{12}\text{C}$ masses $M_c(t)$ vary in time (e.g. post-1850 industrial Suess dilution or glacial carbon-cycle reorganizations) or nonlinear kinetic extensions are introduced, the generator $A(t)$ becomes non-autonomous and `ticktack 2.0` provides two adaptive differential-equation backends:

- **`DiffraxSolver`**: Wraps [`diffrax.diffeqsolve`](https://docs.kidger.site/diffrax/) with explicit `Tsit5` (5th-order Tsitouras Runge–Kutta) for non-stiff/moderate systems or implicit `Kvaerno5` (5th-order singly diagonally implicit Runge–Kutta) with `PIDController(rtol=1e-8, atol=1e-10)` and `RecursiveCheckpointAdjoint` for exact reverse-mode `jax.grad` differentiation. Jump discontinuities at spike boundaries are registered via `jump_ts` so adaptive steppers hit onset/cessation epochs cleanly without step-rejection chatter.
- **`ModaxSolver`**: Provides a lightweight custom adaptive Dormand–Prince / RK solver with continuous adjoint sensitivities implemented via `@jax.custom_vjp`. When wrapping forward-mode-only (`jax.jvp`) functions inside reverse-mode (`jax.grad`) pipelines such as `optimistix` BFGS or `NumPyro` NUTS, `wrap_jvp_only_scalar_fn` bridges `jvp` tangents into a valid reverse-mode `custom_vjp` gradient rule.

On autonomous LTI benchmarks, `ExactLinearSolver`, `DiffraxSolver`, and `ModaxSolver` agree to $< 10^{-7}\text{‰}$ across all bundled presets.
