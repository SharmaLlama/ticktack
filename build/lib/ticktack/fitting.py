import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import celerite2.jax
from celerite2.jax import terms as jax_terms
import jax.numpy as jnp
from jax import grad, jit, partial
import ticktack
from astropy.table import Table
from tqdm import tqdm
import emcee
from chainconsumer import ChainConsumer
import scipy
import seaborn as sns

rcParams['figure.figsize'] = (16.0, 8.0)

class CarbonFitter():
    """
    A class for parametric and non-parametric inference of d14c data using a Carbon Box Model (cbm).
    Does optimization, MCMC sampling, plotting and more.
    """
    def __init__(self, cbm, production_rate_units='atoms/cm^2/s',target_C_14=707.):
        if isinstance(cbm, str):
            try:
                if cbm in ['Guttler14', 'Brehm21', 'Miyake17', 'Buntgen18']:
                    cbm = ticktack.load_presaved_model(cbm, production_rate_units=production_rate_units)
                else:
                    cbm = load_model(cbm, production_rate_units=production_rate_units)
            except:
                raise ValueError('Must be a valid CBM model')
        self.cbm = cbm
        self.cbm.compile()
        self.steady_state_production = self.cbm.equilibrate(target_C_14=target_C_14) # 707 default for Guttler
        self.steady_state_y0 = self.cbm.equilibrate(production_rate=self.steady_state_production)

    def load_data(self, file_name, resolution=1000, fine_grid=0.05, time_oversample=1000, num_offset=4):
        data = Table.read(file_name, format="ascii")
        self.time_data = jnp.array(data["year"])
        self.d14c_data = jnp.array(data["d14c"])
        self.d14c_data_error = jnp.array(data["sig_d14c"])
        self.start = np.nanmin(self.time_data)
        self.end = np.nanmax(self.time_data)
        self.resolution = resolution
        self.burn_in_time = jnp.linspace(self.start-1000, self.start, self.resolution)
        self.time_grid_fine = jnp.arange(self.start, self.end, fine_grid)
        self.time_oversample = time_oversample
        self.offset = jnp.mean(self.d14c_data[:num_offset])
        self.annual = jnp.arange(self.start, self.end + 1)
        self.mask = jnp.in1d(self.annual, self.time_data)[:-1]

    def prepare_function(self, **kwargs):
        self.production = None
        self.gp = None

        try:
            custom_function = kwargs['custom_function']
            try:
                f = kwargs['f']
            except:
                f = None
        except:
            custom_function = False

        try:
            use_control_points = kwargs['use_control_points']
            try:
                interp = kwargs['interp']
            except:
                interp = 'gp'

            try:
                dense_years = kwargs['dense_years']
            except:
                dense_years = 3
            try:
                gap_years = kwargs['gap_years']
            except:
                gap_years = 5
        except:
            use_control_points = False

        try:
            production = kwargs['production']
            try:
                fit_solar_params = kwargs['fit_solar']
            except:
                fit_solar_params = None
        except:
            production = None

        if production == 'miyake':
            if fit_solar_params:
                self.production = self.miyake_event_flexible_solar
            else:
                self.production = self.miyake_event_fixed_solar

        if custom_function is True and f is not None:
            self.production = f

        if use_control_points is True:
            if interp == "linear":
                control_points_time = [self.start - 1, self.start]

                n = len(self.time_data[:-1])
                for i in range(n):
                    if (self.time_data[i] - control_points_time[-1]<=2) & (self.time_data[i] - control_points_time[-1]>0):
                        control_points_time.append(float(self.time_data[i]) + dense_years)
                    elif self.time_data[i] >= control_points_time[-1] + gap_years:
                        control_points_time.append(float(self.time_data[i]))
                control_points_time = np.array(control_points_time)[control_points_time <= self.end]

                self.control_points_time = jnp.array(control_points_time)
                self.production = self.interp_linear

            elif interp == "gp":
                self.control_points_time = jnp.arange(self.start, self.end)
                self.production = self.interp_gp
                self.gp = True


    @partial(jit, static_argnums=(0,))
    def interp_linear(self, tval, *args):
        control_points = jnp.squeeze(jnp.array(list(args)))
        return jnp.interp(tval, self.control_points_time, control_points)

    @partial(jit, static_argnums=(0,))
    def interp_gp(self, tval, *args):
        """
        A Gaussian Process interpolator

        Parameters
        ----------
        tval : ndarray
            Time sampling of the output interpolation
        args: ndarray|float
            The set of control-points. Can be passed in as ndarray or individual floats. Must have the same size as
            self.control_points_time.

        Returns
        -------
        ndarray
            Interpolation results on tval
        """
        tval = tval.reshape(-1)
        params = jnp.array(list(args)).reshape(-1)
        control_points = params
        mean = params[0]
        kernel = jax_terms.Matern32Term(sigma=2., rho=2.)
        gp = celerite2.jax.GaussianProcess(kernel, self.control_points_time, mean=mean)
        alpha = gp.apply_inverse(control_points)
        Ks = kernel.get_value(tval[:, None] - self.control_points_time[None, :])
        mu = jnp.dot(Ks, alpha)
        mu = (tval > self.start) * mu + (tval <= self.start) * mean
        return mu

    @partial(jit, static_argnums=(0,))
    def sum_interp_gp(self, *args):
        mu = self.interp_gp(self.annual, *args)
        return jnp.sum(mu)

    @partial(jit, static_argnums=(0,))
    def grad_sum_interp_gp(self, *args):
        return grad(self.sum_interp_gp)(*args)

    @partial(jit, static_argnums=(0,))
    def neg_gp_log_likelihood(self, params):
        """
        Computes the negative Gaussian Process log-likelihood for non-parametric inference
        using control-points method.

        Parameters
        ----------
        params : ndarray
            Control-points. First control point is also the mean of the Gaussian Process

        Returns
        -------
        float
            Negative Gaussian Process log-likelihood evaluated on 'params'
        """
        kernel = jax_terms.Matern32Term(sigma=2., rho=2.)
        gp = celerite2.jax.GaussianProcess(kernel, mean=params[0])
        gp.compute(self.control_points_time)
        return -gp.log_likelihood(params)

    @partial(jit, static_argnums=(0,))
    def gp_log_likelihood(self, params):
        control_points = params
        mean = params[0]
        kernel = jax_terms.Matern32Term(sigma=2., rho=2.)
        gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
        gp.compute(self.control_points_time)
        return gp.log_likelihood(control_points)

    @partial(jit, static_argnums=(0,))
    def super_gaussian(self, t, start_time, duration, area):
        middle = start_time+duration/2.
        height = area/duration
        return height*jnp.exp(- ((t-middle)/(1./1.93516*duration))**16.)

    @partial(jit, static_argnums=(0,))
    def miyake_event_fixed_solar(self, t, *args):
        start_time, duration, phase, area = list(args)
        height = self.super_gaussian(t, start_time, duration, area)
        prod = self.steady_state_production + 0.18 * self.steady_state_production * jnp.sin(2 * np.pi / 11 * t + phase) + height
        return prod

    @partial(jit, static_argnums=(0,))
    def miyake_event_flexible_solar(self, t, *args):
        start_time, duration, phase, area, omega, amplitude = list(args)
        height = self.super_gaussian(t, start_time, duration, area)
        prod = self.steady_state_production + amplitude * self.steady_state_production * jnp.sin(
            omega * t + phase) + height
        return prod

    @partial(jit, static_argnums=(0,))
    def run(self, time_values, y0, params=()):
        burn_in, _ = self.cbm.run(time_values, production=self.production, args=params, y0=y0)
        return burn_in

    @partial(jit, static_argnums=(0, 2))
    def run_D_14_C_values(self, time_out, time_oversample, y0, params=()):
        d_14_c = self.cbm.run_D_14_C_values(time_out, time_oversample,
                                       production=self.production, args=params, y0=y0,
                                       steady_state_solutions=self.steady_state_y0)
        return d_14_c


    @partial(jit, static_argnums=(0,))
    def dc14(self, params=()):
        """
        Predict d14c on the same time sampling as self.time_data

        Parameters
        ----------
        params : ndarray, optional
            Parameters for self.production

        Returns
        -------
        ndarray
            Predicted d14c value
        """
        burn_in = self.run(self.burn_in_time, self.steady_state_y0, params=params)
        d_14_c = self.run_D_14_C_values(self.annual, self.time_oversample, burn_in[-1, :], params=params)
        d_14_c = d_14_c[self.mask]
        return d_14_c + self.offset

    @partial(jit, static_argnums=(0,))
    def dc14_fine(self, params=()):
        """
        Predict d14c on the same time sampling as self.time_grid_fine.

        Parameters
        ----------
        params : ndarray, optional
            Parameters for self.production

        Returns
        -------
        ndarray
            Predicted d14c value
        """
        burn_in = self.run(self.burn_in_time, self.steady_state_y0, params=params)
        data, solution = self.cbm.run(self.time_grid_fine, production=self.production, args=params, y0=burn_in[-1,:])
        d_14_c = self.cbm._to_d14c(data,self.steady_state_y0)
        return d_14_c + self.offset

    @partial(jit, static_argnums=(0,))
    def log_like(self, params=()):
        """
        Computes the gaussian log-likelihood for parameters of self.production, using the predicted
        d14c values and observed d14c values.

        Parameters
        ----------
        params : ndarray, optional
            Parameters for self.production

        Returns
        -------
        float
            Gaussian log-likelihood
        """
        d_14_c = self.dc14(params=params)
        chi2 = jnp.sum(((self.d14c_data[:-1] - d_14_c)/self.d14c_data_error[:-1])**2)
        return -0.5*chi2

    @partial(jit, static_argnums=(0,))
    def neg_log_like(self, params=()):
        return -1*self.log_like(params=params)

    @partial(jit, static_argnums=(0,))
    def neg_gp_likelihood(self, params=()):
        return self.neg_log_like(params=params) + self.neg_gp_log_likelihood(params)

    @partial(jit, static_argnums=(0,))
    def grad_neg_gp_likelihood(self, params=()):
        return grad(self.neg_gp_likelihood)(params=params)

    @partial(jit, static_argnums=(0,))
    def gp_likelihood(self, params=()):
        return self.log_like(params=params) + self.gp_log_likelihood(params)

    def fit_cp(self, low_bound=0):
        """
        Optimization of control-points for non-parametric inference.

        Parameters
        ----------
        low_bound : int, optional
            The minimum value each control-point can take

        Returns
        -------
        OptimizeResult
            Scipy OptimizeResult object
        """
        initial = self.steady_state_production * jnp.ones((len(self.control_points_time),))
        bounds = tuple([(low_bound, None)] * len(initial))

        if self.gp:
            soln = scipy.optimize.minimize(self.neg_gp_likelihood, initial, bounds=bounds,
                                           options={'maxiter': 20000})
        else:
            soln = scipy.optimize.minimize(self.neg_log_like, initial, bounds=bounds,
                                           method="L-BFGS-B", options={'maxiter': 20000})
        return soln


    def sampling(self, params, likelihood, burnin=500, production=1000):
        """
        Runs an affine-invariant MCMC sampler on an array of initial parameters, subject to
        some likelihood function.

        Parameters
        ----------
        params : ndarray
            Initial parameters for the MCMC
        likelihood : function
            The likelihood function for params
        burnin : int, optional
            Number of steps to run in burn-in period
        production : int, optional
            Number of steps to run in production period

        Returns
        -------
        ndarray
            A chain of MCMC walk
        """
        initial = params
        ndim, nwalkers = len(initial), 5*len(initial)
        if self.gp:
            ndim, nwalkers = len(initial), 2*len(initial)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood)

        print("Running burn-in...")
        p0 = initial + 1e-5 * np.random.rand(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, burnin, progress=True);

        print("Running production...")
        sampler.reset()
        sampler.run_mcmc(p0, production, progress=True);
        return sampler.flatchain

    def plot_recovery(self, chain, time_data=None, true_production=None):
        """
        Takes a chain of MCMC walk, plots random samples from the chain and the true.

        Parameters
        ----------
        chain : ndarray
            The chain of an MCMC walk

        Returns
        -------
        figure
            plot of samples
        """
        mean = np.mean(chain, axis=0)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10), sharex=True)
        n = 100
        top_n = np.random.permutation(len(chain))[:n]
        ax1.errorbar(self.time_data[:-1], self.d14c_data[:-1], yerr=self.d14c_data_error[:-1],
                     fmt="o", color="k", fillstyle="full", capsize=3, markersize=4, label="true d14c")
        for i in tqdm(top_n):
            d14c = self.dc14_fine(params=chain[i, :])
            ax1.plot(self.time_grid_fine, d14c, color="g", alpha=0.2)
            control_points_fine = self.production(self.time_grid_fine, (chain[i, :],))
            ax2.plot(self.time_grid_fine, control_points_fine, color="g", alpha=0.2)
        control_points_fine = self.production(self.time_grid_fine, (mean,))
        ax2.plot(self.time_grid_fine, control_points_fine, "r", label="sample mean production rate")
        ax1.set_ylabel("$\Delta^{14}$C (‰)");
        ax2.set_ylabel("Production rate ($cm^2s^{-1}$)")
        ax2.set_xlabel("Calendar Year")
        if (true_production is not None) & (time_data is not None):
            ax2.plot(time_data, true_production, 'k', label="true production rate")
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True);
        ax1.legend();

    def chain_summary(self, chain, walkers, figsize=(10, 10), labels=None, distribution=False):
        """
        From a chain of MCMC walk, apply convergence test and plot posterior surfaces, or marginal
        distributions, of the the parameters.

        Parameters
        ----------
        chain : ndarray
            The chain of an MCMC walk
        walker : int
            The number of walkers for the chain
        figsize : tuple, optional
            Output figure size. Should be increasing in the dimension of parameters
        labels : list[str], optional
            Parameter names
        distribution : bool, optional
            If True, plot the marginal distributions of parameters instead of posterior surfaces.
            Recommended for high dimensional parameters.

        Returns
        -------
        figure
            plot of marginal distributions or posterior surfaces
        """
        if labels:
            c = ChainConsumer().add_chain(chain, walkers=walkers, parameters=labels)
        else:
            c = ChainConsumer().add_chain(chain, walkers=walkers)
        gelman_rubin_converged = c.diagnostic.gelman_rubin()
        geweke_converged = c.diagnostic.geweke()
        if gelman_rubin_converged and geweke_converged:
            self.convergence = True
        else:
            self.convergence = False
        print("Convergence: %s" % self.convergence)

        if distribution:
            fig = c.plotter.plot_distributions(figsize=figsize)
        else:
            c.configure(spacing=0.0)
            fig = c.plotter.plot(figsize=figsize)
        return fig


    def plot_samples(self, chain):
        """
        Takes a chain of MCMC walk and plots random samples from the chain.

        Parameters
        ----------
        chain : ndarray
            The chain of an MCMC walk

        Returns
        -------
        figure
            plot of samples
        """
        value = jnp.mean(chain, axis=0)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        for s in chain[np.random.randint(len(chain), size=100)]:
            d_c_14_fine = self.dc14_fine(params=s)
            ax1.plot(self.time_grid_fine, d_c_14_fine, alpha=0.2, color="g")

        d_c_14_coarse = self.dc14(params=value)
        d_c_14_fine = self.dc14_fine(params=value)
        ax1.plot(self.time_grid_fine, d_c_14_fine, color="k")

        ax1.plot(self.time_data[:-1], d_c_14_coarse, "o", color="k", fillstyle="none", markersize=7)
        ax1.errorbar(self.time_data, self.d14c_data,
            yerr=self.d14c_data_error, fmt="o", color="k", fillstyle="full", capsize=3, markersize=7)
        ax1.set_ylabel("$\delta^{14}$C (‰)")
        fig.subplots_adjust(hspace=0.05)

        for s in chain[np.random.randint(len(chain), size=10)]:
            production_rate = self.production(self.time_grid_fine, *s)
            ax2.plot(self.time_grid_fine, production_rate, alpha=0.25, color="g")

        mean_draw = self.production(self.time_grid_fine, *value)
        ax2.plot(self.time_grid_fine, mean_draw, color="k", lw=2)
        ax2.set_ylim(jnp.min(mean_draw)*0.8, jnp.max(mean_draw)*1.1);
        ax2.set_xlabel("Calendar Year (CE)");
        ax2.set_ylabel("Production rate ($cm^2s^{-1}$)");
        # if savefile:
        #     figure.savefig(savefile, bbox_inches='tight')

def scatter_plot(array, figsize=10, square_size=100):
    """
    Makes clear and easily understandable heatmap.

    Parameters
    ----------
    array : ndarray
        n x n matrix for the heatmap.
    figsize : int, optional
        Controls the size of the output figure. Should increase with the size of the array.
        Default at 10.
    square_size: int, optional
        Controls the size of squares in the heatmap. Should decrease with the size of the array.
        Default at 100.

    Returns
    -------
    figure
        heatmap
    """
    n = array.shape[0]
    arr = array.reshape(-1)
    size = np.abs(arr)
    x = np.repeat(np.arange(n), n)
    y = (n - 1) - np.tile(np.arange(n), n)

    plot_grid = plt.GridSpec(1, 10, hspace=0.2, wspace=0.1)
    ax = plt.subplots(figsize=(figsize, figsize))
    ax = plt.subplot(plot_grid[:, :-1])

    n_colors = 256
    palette = sns.diverging_palette(10, 220, n=n_colors)

    if np.min(array) < 0:
        bounds = np.abs(np.max((np.min(array), np.max(array))))
        color_min, color_max = (-bounds, bounds)
    else:
        color_min, color_max = (np.min(array), np.max(array))

    def value_to_color(val):
        val_position = (val - color_min) / (color_max - color_min)
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

    ax.scatter(
        x=x, y=y, s=size*square_size, c=[value_to_color(i) for i in arr], marker='s')
    ax.set_xticks(np.unique(x));
    ax.set_yticks(np.unique(x));

    ax.set_xticklabels(np.unique(x));
    ax.set_yticklabels(reversed(np.unique(x)));

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, (n - 1) + 0.5]);
    ax.set_ylim([-0.5, (n - 1) + 0.5]);

    ax = plt.subplot(plot_grid[:, -1])
    col_x = [0] * len(palette)
    bar_y = np.linspace(color_min, color_max, n_colors)
    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5] * len(palette),
        left=col_x,
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2)
    ax.set_ylim(color_min - 0.001, color_max + 0.001)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([color_min, (color_min + color_max) / 2, color_max])
    ax.yaxis.tick_right()
    return