# Bundled Datasets

`ticktack 2.0` bundles an audited collection of published high-resolution tree-ring $\Delta^{14}\text{C}$ records spanning six confirmed or candidate cosmogenic radiation excursions (`775CE`, `993CE`, `660BCE`, `5259BCE`, `5410BCE`, `7176BCE`) together with the `IntCal20` calibration curve compilation. Every bundled CSV has been audited for **hemispheric routing**, **calendar year-zero numbering**, and **Boreal vs Austral growing-season convention** in `ticktack.data.BUNDLED_DATASET_CATALOGUE`.

---

## Loading Bundled Data

Use `load_bundled_event` to assemble all audited series for a given event into a multi-series `Dataset`, or `load_bundled_series` to load an individual `TreeRingSeries` by its catalogue key:

```python
import ticktack as tt

# Load all 28 package-bundled NH + SH series for the 774–775 CE Miyake event
ds_775 = tt.load_bundled_event("775CE")

# Filter to Southern Hemisphere series only
ds_775_sh = tt.load_bundled_event("775CE", hemisphere="south")

# Load a single audited series by catalogue key
ew_660 = tt.load_bundled_series("660BCE-Ew/Sakurai20_CedarEw.csv")
print(ew_660.name, ew_660.calendar, ew_660.seasonal_weights.bounds)

# Inspect all audited catalogue entries and their metadata
for key, entry in list(tt.BUNDLED_DATASET_CATALOGUE.items())[:5]:
    print(f"{key}: hem={entry.hemisphere}, cal={entry.calendar}, season={entry.growth_season} ({entry.year_convention})")
```

When working inside a full git checkout of the repository, pass `include_repo_notebooks=True` to `load_bundled_event("775CE", include_repo_notebooks=True)` to also include the supplementary CSV records stored under `notebooks/775CE/` (such as `Guttler15_Kauri.csv`, `Miyake12_Cedar.csv`, `Jull14_Bristlecone.csv`, and `Rakowski15_Oak.csv`).

### 774 CE Compilations Used Across Documentation and Notebooks

Different tutorials and analyses use distinct subsets of the 774–775 CE tree-ring records. Always record which compilation is used when reporting Fisher statistics (`d_eff`, `info_bits`), evidences, or posterior summaries:

| Compilation | Loader / Source | Series | Rings | Span (CE) | Where Used |
| :--- | :--- | :---: | :---: | :--- | :--- |
| Bundled `775CE` | `load_bundled_event("775CE")` | 28 | 313 | `760.25–781.25` | `v2/01`, `v2/02`, `v2/03`, `v2/05`, Fisher examples |
| Multi-decade `775CE` | `load_bundled_event("775CE")` + 4 long repo series (`Miyake12_Cedar`, `Rakowski15_Oak`, `Jull14_Bristlecone`, `Guttler15_Kauri`) | 32 | 432 | `760.25–800.67` | `v2/04` (`select_gp_kernel`) |
| Full repo `775CE` | `load_bundled_event("775CE", include_repo_notebooks=True)` | 42 | 531 | `760.25–800.67` | Full combined compilation |
| Legacy `v1` multi-series | All 14 `notebooks/775CE/*.csv` series | 14 | 218 | `760.42–800.67` | `docs_examples/02` |
| Single discovery series | `notebooks/miyake12.csv` (`Miyake12_Cedar`) | 1 | 28 | `760.25–787.75` | `docs_examples/01`, `03`, `05`, `06` |

---

## Audited Dataset Catalogue

### Package-Bundled Datasets (`src/ticktack/data/datasets/`)

| Event (`load_bundled_event`) | Catalogue Directory | Series Count | Hemisphere | `calendar` | `year_convention` | `growth_season` | Primary Provenance |
| :--- | :--- | :---: | :---: | :--- | :--- | :--- | :--- |
| `"775CE"` | `775AD-early-N/` | 11 | `"north"` | `"astronomical_ce"` | `"schulman_start"` | `"april-september"` | [Büntgen et al. (2018)](https://doi.org/10.1038/s41467-018-06036-0) |
| `"775CE"` | `775AD-late-N/` | 11 | `"north"` | `"astronomical_ce"` | `"schulman_start"` | `"april-september"` | [Büntgen et al. (2018)](https://doi.org/10.1038/s41467-018-06036-0); [Scifo et al. (2019)](https://doi.org/10.1038/s41598-019-53296-x) |
| `"775CE"` | `775AD-early-S/` | 6 | `"south"` | `"astronomical_ce"` | `"schulman_start"` | `"october-march"` | [Büntgen et al. (2018)](https://doi.org/10.1038/s41467-018-06036-0) (ring `774` $=$ Oct 774–Mar 775 CE) |
| `"993CE"` | `993AD-N/` | 7 | `"north"` | `"astronomical_ce"` | `"schulman_start"` | `"april-september"` | [Büntgen et al. (2018)](https://doi.org/10.1038/s41467-018-06036-0) |
| `"993CE"` | `993AD-S/` | 2 | `"south"` | `"astronomical_ce"` | `"schulman_start"` | `"october-march"` | [Büntgen et al. (2018)](https://doi.org/10.1038/s41467-018-06036-0) (ring `993` $=$ Oct 993–Mar 994 CE) |
| `"660BCE"` | `660BCE-Ew/` | 1 | `"north"` | `"historical_bce"` | `"schulman_start"` | `"april-june"` | [Sakurai et al. (2020)](https://doi.org/10.1038/s41598-020-57658-9) Yakushima cedar earlywood |
| `"660BCE"` | `660BCE-Lw/` | 1 | `"north"` | `"historical_bce"` | `"schulman_start"` | `"july-september"` | [Sakurai et al. (2020)](https://doi.org/10.1038/s41598-020-57658-9) Yakushima cedar latewood |
| `"5259BCE"` | `5259BCE/` | 4 | `"north"` | `"historical_bce"` | `"schulman_start"` | `"april-september"` | [Brehm et al. (2021)](https://doi.org/10.1038/s41561-020-00674-0) (`-5259` $\mapsto$ astro `-5258.0`) |
| `"5410BCE"` | `5410BCE/` | 3 | `"north"` | `"historical_bce"` | `"schulman_start"` | `"april-september"` | [Miyake et al. (2021)](https://doi.org/10.1029/2021GL093419) (`-5410` $\mapsto$ astro `-5409.0`) |
| `"7176BCE"` | `7176BCE/` | 3 | `"north"` | `"historical_bce"` | `"schulman_start"` | `"april-september"` | [Brehm et al. (2021)](https://doi.org/10.1038/s41561-020-00674-0) (`-7176` $\mapsto$ astro `-7175.0`) |
| `"IntCal20"` | `IntCal20/` | 1 | `"north"` | `"astronomical_ce"` | `"schulman_start"` | `"january-december"` | [Reimer et al. (2020)](https://doi.org/10.1017/RDC.2020.41) (`1950 - cal BP`, sorted on load) |

### Supplementary Repository Records (`notebooks/775CE/`)

| Catalogue Key | Hemisphere | `calendar` | `year_convention` | `growth_season` | Audit Notes & Provenance |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `notebooks/775CE/Guttler15_Kauri.csv` | `"south"` | `"astronomical_ce"` | **`"end_year"`** | `"october-march"` | [Güttler et al. (2015)](https://doi.org/10.1016/j.epsl.2014.11.048) New Zealand kauri; ring `775` spans Oct 774–Mar 775 CE ($774.75$–$775.25$), matching the physical time of Schulman ring `774` in `Büntgen18_TAS01.csv`. |
| `notebooks/775CE/Jull14_Bristlecone.csv` | `"north"` | `"astronomical_ce"` | `"schulman_start"` | `"june-august"` | [Jull et al. (2014)](https://doi.org/10.1002/2014GL059874) high-altitude California bristlecone pine. |
| `notebooks/775CE/Miyake12_Cedar.csv` | `"north"` | `"astronomical_ce"` | `"schulman_start"` | `"april-september"` | [Miyake et al. (2012)](https://doi.org/10.1038/nature11123) Japanese cedar discovery series. |
| `notebooks/775CE/Rakowski15_Oak.csv` | `"north"` | `"astronomical_ce"` | `"schulman_start"` | `"april-september"` | [Rakowski et al. (2015)](https://doi.org/10.1016/j.nimb.2015.05.031) Polish oak series. |

---

## CSV File Format for Custom Datasets

`TreeRingSeries.from_csv` and `Dataset.from_csvs` parse UTF-8 CSV files using Python's standard-library `csv` module (no `pandas` dependency). Recognized column headers (case-insensitive):

- `year` *(required)*: Nominal ring year in the declared `calendar`.
- `d14c` *(required)*: Measured age-corrected $\Delta^{14}\text{C}$ in per mil ($\text{‰}$).
- `sig_d14c` (or `sigma_d14c` / `d14c_err`, *required*): $1\sigma$ measurement uncertainty in per mil ($\text{‰}$, strictly positive).
- `growth_season` *(optional)*: Month-range string such as `"april-september"` if not passed via the `growth_season` keyword argument.
- `t_start`, `t_end` *(required only when `year_convention="explicit_bounds"`)*: Exact per-ring start and end times on the continuous astronomical CE axis.

---

## Data Provenance References

- **Brehm et al. (2021)** – Brehm, N., Bayliss, A., Christl, M., Synal, H.-A., Adolphi, F., Beer, J., Kromer, B., Muscheler, R., Solanki, S. K., Usoskin, I., Bleicher, N., Bollhalder, S., Tyers, C., & Wacker, L. *Eleven-year solar cycles over the last millennium revealed by radiocarbon in tree rings*. Nature Geoscience, 14, 10–15 (2021); and Brehm, N. et al. *Tree-rings reveal two strong solar proton events in 7176 and 5259 BCE*. Nature Communications, 13, 1196 (2022).
- **Büntgen et al. (2018)** – Büntgen, U. et al. *Tree rings indicate global environmental downturns that could have been caused by comet or asteroid impacts* / *Tree rings reveal globally coherent signature of cosmogenic radiocarbon events in 774 and 993 CE*. Nature Communications, 9, 3605 (2018).
- **Güttler et al. (2015)** – Güttler, D., Adolphi, F., Beer, J., Bleicher, N., Boswijk, G., Christl, M., Hogg, A., Palmer, J., Vockenhuber, C., Wacker, L., & Wunder, J. *Rapid increase in cosmogenic $^{14}\text{C}$ in AD 775 measured in New Zealand kauri trees indicates short-lived increase in $^{14}\text{C}$ production rate*. Earth and Planetary Science Letters, 411, 290–297 (2015).
- **Jull et al. (2014)** – Jull, A. J. T. et al. *Excursions in the $^{14}\text{C}$ record at A.D. 774–775 in tree rings from Russia and America*. Geophysical Research Letters, 41, 3004–3010 (2014).
- **Miyake et al. (2012, 2021)** – Miyake, F., Nagaya, K., Masuda, K., & Nakamura, T. *A signature of cosmic-ray increase in AD 774–775 from tree rings in Japan*. Nature, 486, 240–242 (2012); and Miyake, F. et al. *More than tenfold increase in cosmic ray flux at 5410 BCE*. Geophysical Research Letters, 48, e2021GL093419 (2021).
- **Reimer et al. (2020)** – Reimer, P. J. et al. *The IntCal20 Northern Hemisphere radiocarbon age calibration curve (0–55 cal kBP)*. Radiocarbon, 62(4), 725–757 (2020).
- **Sakurai et al. (2020)** – Sakurai, H. et al. *Prolonged production of $^{14}\text{C}$ during the ~660 BCE solar proton event from Japanese tree rings*. Scientific Reports, 10, 660 (2020).
- **Scifo et al. (2019)** – Scifo, A. et al. *Radiocarbon production and environmental impacts of the AD 774–775 event in Lithuanian oak*. Scientific Reports, 9, 17056 (2019).
