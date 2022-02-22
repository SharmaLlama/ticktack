import matplotlib.pyplot as plt
import numpy as np

chains = []
for i in range(len(snakemake.input)):
    chain = np.load(snakemake.input[i])
    chains.append(chain)

if snakemake.params.production_model == "simple_sinusoid":
    labels = ["start date (yr)", "duration (yr)", "$\phi$ (yr)", "spike production (atoms/cm$^2$ yr/s)"]
elif snakemake.params.production_model == "flexible_sinusoid":
    labels = ["start date (yr)", "duration (yr)", "$\phi$ (yr)", "spike production (atoms/cm$^2$ yr/s)", "solar amplitude (atoms/cm$^2$/s)"]
elif snakemake.params.production_model == "flexible_sinusoid_affine_variant":
    labels = ["gradient (atoms/cm$^2$/year$^2$)", "start date (yr)", "duration (yr)", "$\phi$ (yr)", "spike production (atoms/cm$^2$ yr/s)", "solar amplitude (atoms/cm$^2$/s)"]
else:
    labels = range(chain.shape[0])

fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
axs = axs.flatten()
for i in range(chain.shape[1]):
    axs[i].plot(chain[:, i], 'b.', markersize=1, alpha=0.5)
    axs[i].set_title(labels[i])
    axs[i].get_xaxis().set_visible(False)
for i in range(chain.shape[1], 6):
    axs[i].set_axis_off()
plt.suptitle(snakemake.params.event + ' -- ' + snakemake.params.cbm_model, fontsize=25)
fig.savefig(snakemake.output[0])
