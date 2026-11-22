# Production Terms (`ticktack.production`)

Composable cosmic-ray $^{14}\text{C}$ production models (`ProductionTerm` subclasses). Terms combine additively via `+` into a `ProductionSum`, allowing parametric Miyake spikes (`MiyakeSpike`), multi-harmonic solar cycles (`HarmonicSolar`), secular trends (`LinearTrend`), gridded templates (`GriddedProduction`), and Solin–Särkkä Hilbert-space Gaussian processes (`GPProduction`, `LogGPProduction`) to be mixed freely inside `RadiocarbonModel`.

---

::: ticktack.production
    options:
      show_root_heading: false
