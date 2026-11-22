# Physical & Calendar Conventions

Precision radiocarbon inference at $\sim 1\text{‰}$ accelerator mass spectrometry (AMS) accuracy requires exact agreement on radioactive decay constants, unit conversions, reference specific activities, baseline offsets, and tree-ring dating conventions. Subtle mismatches in any of these choices introduce systematic biases comparable to or larger than the laboratory error bars themselves. This page documents every physical and calendar convention enforced across `ticktack 2.0`.

---

## 1. Governing LTI System, Units, & Reference Specific Activity

### Coupled Reservoir Mass-Balance ODE

Consider an $n$-box carbon-cycle model with reservoir $^{12}\text{C}$ masses $M_1, \dots, M_n$ (in gigatonnes of carbon, $\text{GtC} = 10^{12}\text{ kg}$) and $^{14}\text{C}$ inventories $\mathbf{y}(t) = (y_1(t), \dots, y_n(t))^\top$ (in $\text{kg}$ of $^{14}\text{C}$). Under linear donor-controlled exchange with first-order rate $k_{i \to j} \ge 0$ ($\text{yr}^{-1}$) from reservoir $i$ to reservoir $j$, radioactive decay rate $\lambda$ ($\text{yr}^{-1}$), and normalized cosmogenic production partition vector $\mathbf{b} \in \Delta^{n-1}$ ($\sum_{i=1}^n b_i = 1$, $b_i \ge 0$), the $^{14}\text{C}$ inventory evolves according to the linear time-invariant (LTI) system

$$
\frac{\mathrm{d}\mathbf{y}(t)}{\mathrm{d}t} = A\,\mathbf{y}(t) + \alpha_{\text{unit}}\,Q(t)\,\mathbf{b},
$$

where $Q(t)$ is the global-mean surface cosmogenic $^{14}\text{C}$ production rate in $\text{atoms cm}^{-2}\text{ s}^{-1}$ and the system generator $A \in \mathbb{R}^{n \times n}$ has entries

$$
A_{ij} = \begin{cases}
k_{j \to i}, & i \neq j, \\[4pt]
-\displaystyle\sum_{\ell \neq i} k_{i \to \ell} - s_i - \lambda, & i = j,
\end{cases}
$$

with $s_i \ge 0$ ($\text{yr}^{-1}$) denoting any irreversible sediment-burial or open-boundary sink rate out of box $i$ ($s_i = 0$ in closed `BalancedCBM` and `ReversibleCBM` models). Because every column sum of the conservative transport operator is zero and $\lambda > 0$, $A$ is strictly Metzler–Hurwitz (all eigenvalues satisfy $\operatorname{Re}\lambda_k(A) \le -\lambda < 0$), guaranteeing a unique globally stable steady state

$$
\mathbf{y}_{\text{ss}}(Q_{\text{std}}) = -\alpha_{\text{unit}}\,Q_{\text{std}}\,A^{-1}\mathbf{b}.
$$

### Decay Constants & Production Unit Conversion

All time derivatives and transfer rates in `ticktack 2.0` are expressed per **Julian year** ($365.25 \times 86\,400 = 31\,557\,600\text{ s yr}^{-1}$, `SECONDS_PER_JULIAN_YEAR`). Three radiometric decay constants ($\text{yr}^{-1}$) are provided in `ticktack.constants`:

| Constant | Definition | Value ($\text{yr}^{-1}$) | Role |
| :--- | :--- | :--- | :--- |
| `DECAY_CAMBRIDGE` | $\ln 2 / 5730$ | $1.20968094 \times 10^{-4}$ | **Default physical decay rate** (`DEFAULT_DECAY_RATE`) used in forward transport (`Godwin 1962`). |
| `DECAY_KUTSCHERA` | $\ln 2 / 5700$ | $1.21604768 \times 10^{-4}$ | Updated half-life assessment (`Kutschera 2013`). |
| `DECAY_STUIVER_POLACH` | $1 / 8267$ | $1.20962864 \times 10^{-4}$ | Conventional mean-life constant in Stuiver & Polach (1977) age-reporting formulas ($t_{1/2} = 5568\text{ yr}$). |

The CODATA / IUGG conversion factor `ALPHA_ATOMS_CM2_S_TO_KG_YR` maps $1\text{ atom cm}^{-2}\text{ s}^{-1}$ averaged over Earth's mean sphere ($R_\oplus = 6.3710088 \times 10^8\text{ cm}$, $4\pi R_\oplus^2 = 5.1006562 \times 10^{18}\text{ cm}^2$) with $^{14}\text{C}$ atomic mass $m_{14} = 14.0032419884\text{ g mol}^{-1}$ (AME2020) and Avogadro constant $N_{\text{A}} = 6.02214076 \times 10^{23}\text{ mol}^{-1}$:

$$
\alpha_{\text{unit}} = \frac{m_{14} \times 10^{-3}}{N_{\text{A}}}\,\left(4\pi R_\oplus^2\right)\,\left(365.25 \times 86\,400\right) \approx 3.743832 \text{ kg yr}^{-1}\;\big/\;(\text{atom cm}^{-2}\text{ s}^{-1}).
$$

The standard pre-industrial reference production rate is `DEFAULT_Q_STD = 1.76` $\text{atoms cm}^{-2}\text{ s}^{-1}$ ([Güttler et al. 2015](https://doi.org/10.1016/j.epsl.2014.11.048)), corresponding to $\alpha_{\text{unit}} Q_{\text{std}} \approx 6.589\text{ kg yr}^{-1}$ of $^{14}\text{C}$ globally. For exact bit-for-bit comparison with `ticktack 1.x` scripts (which used a $365$-day year and rounded $4\pi R_\oplus^2 = 5.11 \times 10^{18}\text{ cm}^2$), `ALPHA_LEGACY_V1` is retained in `ticktack.constants`.

### Single Global Tropospheric Reference Activity $R_{\text{ref}}$

Age-corrected $\Delta^{14}\text{C}$ (in per mil, $\text{‰}$; [Stuiver & Polach 1977](https://doi.org/10.1017/S0033822200003672)) measures the fractional departure of a reservoir's specific activity $R_c(t) = y_c(t) / M_c$ from an international pre-industrial oxalic-acid standard $R_{\text{ref}}$:

$$
\Delta^{14}\text{C}_c(t) = 1000\,\left(\frac{y_c(t) / M_c}{R_{\text{ref}}} - 1\right).
$$

!!! warning "Preserving the Inter-Hemispheric $\Delta^{14}\text{C}$ Gradient"
    In two-hemisphere models such as `Brehm21`, upwelling of $^{14}\text{C}$-depleted circumpolar deep water in the Southern Ocean depresses Southern Hemisphere tropospheric $^{14}\text{C}/^{12}\text{C}$ relative to the Northern Hemisphere by **$\approx 4.38\text{‰}$** in steady state ([Hogg et al. 2020](https://doi.org/10.1017/RDC.2020.59); [Brehm et al. 2021](https://doi.org/10.1038/s41561-020-00674-0)). Normalizing each hemisphere separately to its own steady-state box inventory ($y_c(t)/y_{c,\text{ss}} - 1$, as in `ticktack 1.x`) artificially erases this $4.38\text{‰}$ NH–SH offset!

`ticktack 2.0` defines a **single global tropospheric reference specific activity** $R_{\text{ref}}$ (`CarbonBoxModel.reference_specific_activity`) as the mass-weighted steady-state mean over all tropospheric boxes $\mathcal{T}$:

$$
R_{\text{ref}} = \frac{\sum_{c \in \mathcal{T}} y_{c,\text{ss}}(Q_{\text{std}})}{\sum_{c \in \mathcal{T}} M_c}.
$$

In a single-troposphere model (`Guttler15`, `Miyake17`, `EffectiveCBM`), $|\mathcal{T}| = 1$ so $\Delta^{14}\text{C}_{\text{trop,ss}} = 0\text{‰}$ identically. In `Brehm21` ($M_{\text{NH}} = M_{\text{SH}}$), the global mean sits midway between the two hemispheres ($\Delta^{14}\text{C}_{\text{NH,ss}} \approx +2.19\text{‰}$, $\Delta^{14}\text{C}_{\text{SH,ss}} \approx -2.19\text{‰}$), preserving the physical $4.38\text{‰}$ inter-hemispheric gradient across all joint NH+SH fits.

---

## 2. Baseline Offsets, Holocene Yield Conventions, & Suess Dilution

### Hierarchical Baseline & Laboratory Offsets

Historical Miyake events occur on top of multi-millennial geomagnetic dipole and ocean-circulation variations, so the ambient tropospheric baseline $\Delta_0$ at epoch $t_0$ rarely equals $0\text{‰}$ (for example, $\Delta_0 \approx -22\text{‰}$ around 774 CE and $\Delta_0 \approx +85\text{‰}$ around 5410 BCE). Furthermore, independent AMS laboratories exhibit $\sim 1$–$2\text{‰}$ inter-laboratory calibration offsets ([Jull et al. 2014](https://doi.org/10.1002/2014GL059874); [Büntgen et al. 2018](https://doi.org/10.1038/s41467-018-06036-0)).

If the long-term ambient production rate prior to an event is scaled by $(1 + \Delta_{\text{global}} / 1000)$, linearity of $A^{-1}\mathbf{b}$ implies that box $c$'s steady-state specific activity shifts by $\Delta_{\text{global}}\,r_c$, where $r_c = R_{c,\text{ss}} / R_{\text{ref}}$ ($r_c = 1$ in 1-hemisphere models; $r_{\text{NH}} \approx 1.00219$, $r_{\text{SH}} \approx 0.99781$ in `Brehm21`). Including per-series laboratory offsets $\delta_{\text{lab},s}$, the predicted ring observable for series $s$ routed to box $c(s)$ is

$$
\Delta^{14}\text{C}_{\text{pred},s}(Y) = \mathcal{B}_{\text{season},s}\!\left[\Delta^{14}\text{C}_{c(s)}\right](Y) + \Delta_{\text{global}}\,r_{c(s)} + \delta_{\text{lab},s}.
$$

In `LinearMarginalConfig(offset_mode="hierarchical")`, `ticktack 2.0` places independent Gaussian priors $\Delta_{\text{global}} \sim \mathcal{N}(\mu_{\text{global}}, \sigma_{\text{global}}^2)$ (default $\sigma_{\text{global}} = 200\text{‰}$) and $\delta_{\text{lab},s} \sim \mathcal{N}(0, \sigma_{\text{lab}}^2)$ (default $\sigma_{\text{lab}} = 1.5\text{‰}$) and marginalizes all $S + 1$ offset parameters analytically – completely replacing the biased `ticktack 1.x` practice of clamping the model baseline to the sample mean of the first four rings (`mean(y[:4])`).

### `relative_to="standard"` vs `"ambient"` Yield Conventions

Literature studies report Miyake-event cosmic-ray fluences either in absolute physical units ($\text{atoms cm}^{-2}$) or as dimensionless multiples of one year of steady background production ($q_0\text{-years}$). Across Holocene geomagnetic excursions, $q_0\text{-years}$ becomes ambiguous unless one specifies *which* steady production rate serves as the unit (`relative_to` on `AbstractSolver.predict_d14c` and `baseline_d14c_permil` on `RadiocarbonModel.fluence_q0_years`):

- **Pre-industrial standard (`relative_to="standard"`, default on `solver.predict_d14c`)**: `MiyakeSpike.fluence` is parametrized in absolute units of $\text{atoms cm}^{-2}\text{ s}^{-1}\text{ yr}$, so the physical surface fluence is independent of ambient $\Delta_0$:
  $$
  \mathcal{F}_{\text{phys}} = \text{fluence} \times 31\,557\,600\;\text{atoms cm}^{-2} \quad\text{(`RadiocarbonModel.fluence_atoms_cm2`)}.
  $$
  Dividing by $Q_{\text{std}} = 1.76\text{ atoms cm}^{-2}\text{ s}^{-1}$ gives pre-industrial $q_0\text{-years}$ (`RadiocarbonModel.fluence_q0_years(fluence)`).
- **Epoch-local ambient (`relative_to="ambient"` on `solver.predict_d14c`, or `baseline_d14c_permil=lp.baseline_d14c` in `RadiocarbonModel.fluence_q0_years`)**: When ambient Holocene $\Delta^{14}\text{C}_0 \neq 0\text{‰}$ reflects a quasi-steady geomagnetic modulation $Q_{\text{amb}} = Q_{\text{std}}(1 + \Delta_{\text{global}} / 1000)$, `relative_to="ambient"` multiplies excursions by $(1 + \Delta_{\text{global}} / 1000)$ so `fluence` is measured relative to the ambient epoch production rate, and `fluence_q0_years(fluence, baseline_d14c_permil=Delta_0)` converts standard `fluence` to epoch-local $q_0\text{-years}$:
  $$
  \tau_{q_0,\text{epoch}} = \frac{\text{fluence}}{Q_{\text{std}}\,(1 + \Delta_0 / 1000)}\;\text{yr}.
  $$
  For mid-Holocene events such as 5259 BCE or 5410 BCE ($\Delta_0 \approx +70\text{ to }+90\text{‰}$), pre-industrial and epoch-local $q_0\text{-years}$ differ by nearly $8$–$9\%$! Always report $\mathcal{F}_{\text{phys}}$ in $\text{atoms cm}^{-2}$ alongside the explicit $q_0$ reference convention.

### Non-Autonomous Suess Dilution via $M_c(t)$

Post-1850 fossil-fuel combustion injects $^{14}\text{C}$-free ("dead") carbon into the atmosphere, expanding tropospheric $^{12}\text{C}$ mass $M_{\text{trop}}(t)$ and depressing $\Delta^{14}\text{C}(t)$ (the Suess effect). In non-autonomous mode (`DiffraxSolver` or `ModaxSolver`), `CarbonBoxModel` accepts a time-dependent mass trajectory $M_c(t)$: donor-controlled $^{14}\text{C}$ outflux rates scale dynamically as $F_{i \to j}(t) / M_i(t)$ and specific activity is evaluated against the instantaneous reservoir mass $y_c(t) / M_c(t)$ normalized by the pre-industrial $R_{\text{ref}}$.

---

## 3. Mandatory Calendar & Hemispheric Ring-Dating Conventions

### Continuous Astronomical CE Internal Time Axis

All internal solver grids, spike onset times (`t_start`), and growth windows in `ticktack 2.0` live on the **continuous ISO 8601 / astronomical Common Era (`"astronomical_ce"`)** real line (`float64`), where fractional values mark elapsed Julian-year fractions since 1 January 00:00 UTC. `convert_calendar_years` and `TreeRingSeries` enforce exact conversion across the three standard dendrochronological calendars:

| `calendar` | Year `0` exists? | Mapping to internal `astronomical_ce` | Example (`5259 BCE` Miyake event) |
| :--- | :--- | :--- | :--- |
| `"astronomical_ce"` | Yes (`0.0` $= 1\text{ BCE}$) | Identity: $t_{\text{CE}} = Y$ | `Y = -5258` $\mapsto -5258.0$ |
| `"cal_bp"` | Yes (`0 cal BP` $= 1950.0\text{ CE}$) | $t_{\text{CE}} = 1950.0 - Y_{\text{BP}}$ | `Y_BP = 7208` $\mapsto -5258.0$ |
| `"historical_bce"` | **No** (`ValueError` on `0`) | $t_{\text{CE}} = \begin{cases} Y, & Y > 0 \\ Y + 1, & Y < 0 \end{cases}$ | `Y = -5259` ($5259\text{ BCE}$) $\mapsto -5258.0$ |

!!! danger "The 1-Year Historical BCE Trap"
    Published BCE CSV tables frequently encode $660\text{ BCE}$ as `-660` or $5259\text{ BCE}$ as `-5259` under traditional Dionysian BC/AD numbering, which jumps directly from `1 BCE` (`-1`) to `1 CE` (`+1`) with no year zero. Passing negative historical BCE labels without `calendar="historical_bce"` shifts every BCE ring and inferred spike onset by **exactly 1.0 year** relative to astronomical ephemerides and `cal BP` ice-core chronologies!

### Boreal vs Austral Growth Seasons & Schulman Dating

Tree-ring cellulose records photosynthetically fixed tropospheric $^{14}\text{CO}_2$ integrated over the active cambial growing season:

$$
\mathcal{B}_{\text{season}}[f](Y) = \sum_{m=1}^{M_{\text{sub}}} w_m \left(\frac{1}{b_m - a_m}\int_{Y + a_m}^{Y + b_m} f(t)\,\mathrm{d}t\right), \qquad \sum_{m=1}^{M_{\text{sub}}} w_m = 1.
$$

In the **Northern Hemisphere** (`hemisphere="north"`), cambial growth falls entirely within a single calendar year $Y$ (typically April–September, offsets $[+3/12, +9/12] = [0.25, 0.75]$; earlywood `"april-june"` $[0.25, 0.50]$, latewood `"july-september"` $[0.50, 0.75]$).

In the **Southern Hemisphere** (`hemisphere="south"`), Austral summer growth spans New Year's Day (typically October through March or September through February). Two incompatible ring-labelling conventions coexist in the radiocarbon literature:

1. **`year_convention="schulman_start"`** ([Schulman 1956](https://ltrr.arizona.edu/)): Assigns an Austral ring to the calendar year in which growth **begins** (October of $Y$ through March of $Y + 1$, offsets $[+0.75, +1.25]$). Used in `Büntgen18_TAS01.csv` and `993AD-S`, where the 774 CE event jumps between ring labels `773` and `774`.
2. **`year_convention="end_year"`**: Assigns an Austral ring to the calendar year in which growth **completes** (October of $Y - 1$ through March of $Y$, offsets $[-0.25, +0.25]$, or $[-4/12, +2/12]$ for `"september-february"`). Used in `Guttler15_Kauri.csv` (`775AD-early-S`), where the 774 CE event jumps between ring labels `774` and `775` – representing the **exact same physical Austral summer ($774.67$–$775.17\text{ CE}$)** as Schulman year `774`!
3. **`year_convention="explicit_bounds"`**: Supplies exact per-ring `[t_starts, t_ends]` columns in astronomical CE directly (used for custom sub-annual micro-sliced wood records).

To prevent silent $1\text{-year}$ inter-hemispheric timing artefacts (`ticktack 1.x` treated all Southern Hemisphere CSVs uniformly without distinguishing `schulman_start` from `end_year`), `parse_growth_season_bounds` and `TreeRingSeries` raise a `ValueError` whenever a cross-New-Year season is constructed without an explicit `year_convention`. Arbitrary $M$-bin sub-annual photosynthesis density profiles (e.g. 12 monthly weights) are supported via `SeasonalWeights.from_subannual_weights(...)`.
