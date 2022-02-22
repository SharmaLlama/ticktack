import matplotlib.pyplot as plt
import numpy as np
from ticktack import fitting
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from matplotlib.lines import Line2D
mpl.style.use('seaborn-colorblind')

if snakemake.params.year > 0:
    start_date = "start date ({})".format("yr - " + str(int(snakemake.params.year) - 5) + "CE")
else:
    start_date = "start date ({})".format(str(-int(snakemake.params.year) + 5) + "BCE - yr")
chains = []
for i in range(len(snakemake.input)):
    chain = np.load(snakemake.input[i])
    chains.append(chain)
if snakemake.params.production_model == "simple_sinusoid":
    labels = [start_date, "duration (yr)", "spike production (atoms/cm$^2$ yr/s)", "$\phi$ (yr)"]
    idx = 0
    spike_idx  = 3
elif snakemake.params.production_model == "flexible_sinusoid":
    labels = [start_date, "duration (yr)", "spike production (atoms/cm$^2$ yr/s)", "$\phi$ (yr)", "solar amplitude (atoms/cm$^2$/s)"]
    idx = 0
    spike_idx  = 3
elif snakemake.params.production_model == "flexible_sinusoid_affine_variant":
    labels = ["gradient (atoms/cm$^2$/year$^2$)", start_date, "duration (yr)", "spike production (atoms/cm$^2$ yr/s)", "$\phi$ (yr)", "solar amplitude (atoms/cm$^2$/s)"]
    idx = 1
    spike_idx  = 4
else:
    labels = None

for chain in chains:
    chain[:, idx] = chain[:, idx] - (snakemake.params.year - 5)
    spike = chain.copy()[:, spike_idx]
    chain[:, spike_idx] = chain[:, spike_idx - 1]
    chain[:, spike_idx - 1] = spike

colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
cf = fitting.CarbonFitter()
fig = cf.plot_multiple_chains(chains, chain.shape[1] * 2,
                        params_labels=labels,
                        labels = snakemake.params.cbm_label,
                        label_font_size=7,
                        tick_font_size=7, colors=colors, max_ticks=4, legend=False
                        )
plt.suptitle(snakemake.params.event_label, fontsize=25)
font = font_manager.FontProperties(family='serif', size=14)
custom_lines = [Line2D([0], [0], color=colors[i], lw=0, label=snakemake.params.cbm_label[i]) for i in
                range(len(snakemake.params.cbm_label))]
ax = fig.get_axes()[5]
legend = ax.legend(handles=custom_lines, frameon=False, labelcolor=colors,
                   prop=font, loc="upper right", bbox_to_anchor=(1, 0.5))
fig.savefig(snakemake.output[0])
