import numpy as np
import ticktack
from ticktack import fitting
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from chainconsumer import ChainConsumer

cbm = ticktack.load_presaved_model("Guttler15", production_rate_units = "atoms/cm^2/s")

sf = fitting.SingleFitter(cbm, cbm_model="Guttler15")
sf.load_data("../notebooks/miyake12.csv")
sf.compile_production_model(model="simple_sinusoid")

default_params = np.array([775., np.log10(1./12), np.pi/2., np.log10(81./12)]) # start date, duration, phase, area
sampler = sf.MarkovChainSampler(default_params, 
                                likelihood = sf.log_joint_likelihood, 
                                burnin = 500, 
                                production = 2000, 
                                args = (np.array([770., np.log10(1/52.), 0, -2]), # lower bound
                                np.array([780., np.log10(5.), 11, 1.5]))         # upper bound 
                               )
samples = sampler.copy()
samples[:,1] = 10**samples[:,1] # duration not log duration
samples[:,-1] = 10**samples[:,-1] # area not log area

def chain_summary(sf, chain, walkers, figsize=(10, 10), labels=None, plot_dist=False,
                    label_font_size=8, tick_font_size=8, mle=False,usetex=False):
    """
    Runs convergence tests and plots posteriors from a MCMC chain.
    Parameters
    ----------
    chain : ndarray
        A MCMC chain
    walkers : int
        The total number of walkers of the chain
    figsize : tuple, optional
        Output figure size
    labels : list[str], optional
        A list of parameter names
    plot_dist : bool, optional
        If True, plot the marginal distributions of parameters. Else, plot both the marginal distribution
            and the posterior surface
    """
    c = ChainConsumer().add_chain(chain, walkers=walkers, parameters=labels)

    c.configure(spacing=0.0, usetex=usetex, label_font_size=label_font_size, tick_font_size=tick_font_size,
                diagonal_tick_labels=False,summary=False)
    fig = c.plotter.plot(figsize=figsize)
        
    return fig


labels = ["Start Date (yr)", "Duration (yr)", "φ (yr)", "Production ($q_0$ yr)"]
fig = chain_summary(sf, samples, 8, labels=labels, figsize=(20.0,8.0), label_font_size=10, tick_font_size=10,)
# fig = sf.chain_summary(samples, 8, labels=labels, label_font_size=10, tick_font_size=10,usetex=False)
fig.subplots_adjust(right=0.5)

gs = mpl.gridspec.GridSpec(1,2, width_ratios=[1, 1])

subfig = fig.add_subfigure(gs[0, 1])

(ax1, ax2) = subfig.subplots(2,1, sharex=True,gridspec_kw={'height_ratios': [2, 1]})
# fig.subplots_adjust(hspace=0.05)
plt.rcParams.update({"text.usetex": False})
idx = np.random.randint(len(sampler), size=100)
for param in tqdm(sampler[idx]):
    ax1.plot(sf.time_data_fine, sf.dc14_fine(params=param), alpha=0.05, color="g")

for param in tqdm(sampler[idx][:30]):
    ax2.plot(sf.time_data_fine, sf.production(sf.time_data_fine, *param), alpha=0.2, color="g")

ax1.errorbar(sf.time_data + sf.time_offset, sf.d14c_data, yerr=sf.d14c_data_error, 
             fmt="ok", capsize=3, markersize=6, elinewidth=3, label="$\Delta^{14}$C data")
ax1.legend(frameon=False);
ax2.set_ylim(1, 10);
ax1.set_ylabel("$\Delta^{14}$C (‰)")
ax2.set_xlabel("Calendar Year");
ax2.set_ylabel("Production rate (atoms cm$^2$s$^{-1}$)")

plt.savefig('joss_figure.png',bbox_inches='tight',dpi=300)