import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
import celerite2.jax
from celerite2.jax import terms as jax_terms
import jax.numpy as jnp
from jax import grad, jit, random
from functools import partial
import ticktack
from astropy.table import Table
from tqdm import tqdm
import emcee
from chainconsumer import ChainConsumer
import scipy
import seaborn as sns
from jaxns.nested_sampling import NestedSampler
from jaxns.prior_transforms import PriorChain, UniformPrior
import os


class CarbonFitter:
    """
    Parent class of SingleFitter and MultiFitter. Does Monte Carlo sampling, plotting and more.
    """

    def MarkovChainSampler(self, params, likelihood, burnin=500, production=1000, k=2, args=()):
        """
        Runs an affine-invariant MCMC sampler on an array of initial parameters, subject to some likelihood function.
        Parameters
        ----------
        params : ndarray
            Initial parameters for MC sampler
        likelihood : callable
            Log-likelihood function for params
        burnin : int, optional
            Number of steps to run in burn-in period. 500 by default.
        production : int, optional
            Number of steps to run in production period. 1000 by default.
        k: int, optional
            Determines the number of walkers of the sampler via:
            nwalkers = k * dim(params)
            2 by default.
        Returns
        -------
        ndarray
            A chain of MCMC walk
        """
        initial = params
        ndim, nwalkers = len(initial), k * len(initial)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, args=args)

        print("Running burn-in...")
        p0 = initial + 1e-5 * np.random.rand(nwalkers, ndim)
        p0, lp, _ = sampler.run_mcmc(p0, burnin, progress=True);

        print("Running production...")
        sampler.reset()
        sampler.run_mcmc(p0, production, progress=True);
        return sampler.flatchain

    def NestedSampler(self, params, likelihood, low_bound=None, high_bound=None, sampler_name='multi_ellipsoid'):
        """
        Runs Nested Sampling sampler on the parameter space of some model, subject to some likelihood function.
        Parameters
        ----------
        params : ndarray
            Example parameters for NS sampler
        likelihood : callable
            Log-likelihood function for the set of parameters to be sampled
        low_bound : ndarray, optional
            Lower bound of params
        high_bound : ndarray, optional
            Upper bound of params
        sampler_name : str, optional
            Name of sampling method. Take value in ['multi_ellipsoid', 'slice']. 'multi_ellipsoid' by default.
        Returns
        -------
        ndarray
            A chain of MCMC walk
        """
        @jit
        def likelihood_function(params, **kwargs):
            return likelihood(params)

        ndim = params.size
        if low_bound is not None and high_bound is not None:
            low_bound = jnp.array(low_bound)
            high_bound = jnp.array(high_bound)
        else:
            low_bound = jnp.array(ndim * None)
            high_bound = jnp.array(ndim * None)
        prior_chain = PriorChain().push(UniformPrior('params', low=low_bound, high=high_bound))
        ns = NestedSampler(likelihood_function, prior_chain, num_live_points=100 * prior_chain.U_ndims,
                           sampler_name=sampler_name)
        results = jit(ns)(key=random.PRNGKey(0))
        # summary(results)
        return results

    def chain_summary(self, chain, walkers, figsize=(10, 10), labels=None, plot_dist=False, test_convergence=False):
        """
        From a chain of MCMC walks apply convergence test and plot posterior surfaces of parameters
        Parameters
        ----------
        chain : ndarray
            A chain of MCMC walks
        walkers : int
            The number of walkers for the chain
        figsize : tuple, optional
            Output figure size
        labels : list[str], optional
            A list of parameter names
        plot_dist : bool, optional
            If True, only plot the marginal distributions of parameters
        Returns
        -------
        """
        if labels:
            c = ChainConsumer().add_chain(chain, walkers=walkers, parameters=labels)
        else:
            c = ChainConsumer().add_chain(chain, walkers=walkers)

        if test_convergence:
            gelman_rubin_converged = c.diagnostic.gelman_rubin()
            geweke_converged = c.diagnostic.geweke()
            if gelman_rubin_converged and geweke_converged:
                self.convergence = True
            else:
                self.convergence = False
            print("Convergence: %s" % self.convergence)

        if plot_dist:
            fig = c.plotter.plot_distributions(figsize=figsize)
        else:
            c.configure(spacing=0.0)
            fig = c.plotter.plot(figsize=figsize)

    def correlation_plot(self, array, figsize=10, square_size=100):
        """
        Makes an accessible heatmap for visualizing correlation/covariance matrix.
        Parameters
        ----------
        array : ndarray
            n x n matrix for the heatmap
        figsize : int, optional
            Controls the size of the output figure. Should increase with the size of 'array'. 10 by default
        square_size: int, optional
            Controls the size of squares in the heatmap. Should decrease with the size of 'array'. 100 by default
        Returns
        -------
        figure
            heatmap
        """
        assert array.shape[0] == array.shape[1], "array must be a square (n x n) matrix"
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
            x=x, y=y, s=size * square_size, c=[value_to_color(i) for i in arr], marker='s')
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

    def plot_multiple_chains(self, chains, walker, figsize=(10, 10), title=None, params_names=None, labels=None, colors=None,
                             alpha=0.5, linewidths=None, plot_dists=False):
        """
       Overplots posterior surfaces of parameters from multiple chains.
        Parameters
        ----------
        chains : list
            List of chains of MCMC walks
        walker : int
            The number of walkers for each chain in 'chains'
        figsize : tuple, optional
            Output figure size
        params_names : list[str], optional
            A list of parameter names
        labels : list[str], optional
            Labels that distinguish different chains
        colors : list[str], optional
            A list of color names, used to distinguish different chains
        alpha : float, optional
            Parameter for blending, between 0-1.
        linewidths : float, optional
            Line width, in points
        plot_dists : bool, optional
            If True, only plot the marginal distributions of parameters
        Returns
        -------
        figure
            plot of posterior surfaces or marginal distributions
        """
        c = ChainConsumer()
        if labels:
            assert len(labels) == len(chains), "labels must have the same length as chains"
            for i in range(len(chains)):
                c.add_chain(chains[i], walkers=walker, parameters=params_names, name=labels[i])
        else:
            for i in range(len(chains)):
                c.add_chain(chains[i], walkers=walker, parameters=params_names)
        c.configure(colors=colors, shade_alpha=alpha, linewidths=linewidths)

        if plot_dists:
            fig = c.plotter.plot_distributions(figsize=figsize)
        else:
            fig = c.plotter.plot(figsize=(10, 10))
        plt.suptitle(title)
        plt.tight_layout()
        return fig

    def load_time_data(self, time_data):
        self.time_data = jnp.array(time_data)
        self.start = np.nanmin(self.time_data)
        self.end = np.nanmax(self.time_data)
        self.resolution = resolution
        self.burn_in_time = jnp.linspace(self.start - 1000, self.start, self.resolution)
        self.time_data_fine = jnp.arange(self.start, self.end, fine_grid)
        self.oversample = oversample
        self.offset = jnp.mean(self.d14c_data[:num_offset])
        self.annual = jnp.arange(self.start, self.end + 1)
        self.mask = jnp.in1d(self.annual, self.time_data)

class SingleFitter(CarbonFitter):
    """
    A class for parametric and non-parametric inference of d14c data using a Carbon Box Model (cbm).
    Does parameter fitting, Monte Carlo sampling, plotting and more.
    """

    def __init__(self, cbm, production_rate_units='atoms/cm^2/s', target_C_14=707., box='Troposphere',
                 hemisphere='north'):
        """
        Initializes a SingleFitter Object
        Parameters
        ----------
        cbm : CarbonBoxModel Object
            A Carbon Box Model
        production_rate_units : str, optional
            The production rate units of the cbm. 'atoms/cm^2/s' by default
        target_C_14 : float, optional
            target 14C content for equilibration, 707 by default
        box : str, optional
            The specific box at which to calculate the d14c. 'Troposphere' by default
        hemisphere : str, optional
            The hemisphere which the SingleFitter object will model, can take values in
            ['north', 'south']. 'north' by default
        Returns
        -------
        """
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
        self.steady_state_production = self.cbm.equilibrate(target_C_14=target_C_14)
        self.steady_state_y0 = self.cbm.equilibrate(production_rate=self.steady_state_production)
        self.box = box
        self.hemisphere = hemisphere

    def load_data(self, file_name, resolution=1000, oversample=108, num_offset=4):
        """
        Loads d14c data from specified file
        Parameters
        ----------
        file_name : str
            Path to the file
        resolution : int, optional
            1000 by default
        fine_grid : float, optional
            0.05 by default
        oversample : int, optional
            1000 by default
        num_offset : int, optional
            When set to x the first x data points are averaged to compute an offset, which will be subtracted from
            all data points
        Returns
        -------
        """
        data = Table.read(file_name, format="ascii")
        self.time_data = jnp.array(data["year"])
        self.d14c_data = jnp.array(data["d14c"])
        self.d14c_data_error = jnp.array(data["sig_d14c"])
        self.start = np.nanmin(self.time_data)
        self.end = np.nanmax(self.time_data)
        self.resolution = resolution
        self.burn_in_time = jnp.linspace(self.start - 1000, self.start, self.resolution)
        self.oversample = oversample
        self.time_data_fine = jnp.linspace(self.start - 1, self.end + 1, int(self.oversample * (self.end - self.start + 2)))
        self.offset = jnp.mean(self.d14c_data[:num_offset])
        self.annual = jnp.arange(self.start, self.end + 1)
        self.mask = jnp.in1d(self.annual, self.time_data)
        if self.hemisphere is 'north':
            self.growth = jnp.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        else:
            self.growth = jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])

    def prepare_function(self, model=None):
        """
        Specifies the production rate function
        Parameters
        ----------
        model : str | callable, optional
            Specifies a built-in model or a custom model. Currently supported built-in models include ['simple_sinusoid',
            'flexible_sinusoid', 'control_points']
        Returns
        -------
        """
        self.production = None
        self.gp = None
        if callable(model):
            self.production = model
        elif model == "simple_sinusoid":
            self.production = self.simple_sinusoid
        elif model == "flexible_sinusoid":
            self.production = self.flexible_sinusoid
        elif model == "control_points":
            self.control_points_time = jnp.arange(self.start, self.end)
            self.production = self.interp_gp
            self.gp = True
        else:
            raise ValueError("model is not a callable, or does not take value from: simple_sinusoid, flexible_sinusoid, control_points")

    @partial(jit, static_argnums=(0,))
    def interp_gp(self, tval, *args):
        """
        A Gaussian Process regression interpolator
        Parameters
        ----------
        tval : ndarray
            Time sampling of the output interpolation
        args : ndarray | float
            Set of control-points. Can be passed in as ndarray or individual floats. Must have the same size as
            self.control_points_time.
        Returns
        -------
        ndarray
            Interpolated values on tval
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
    def super_gaussian(self, t, start_time, duration, area):
        """
        Computes the density of a super gaussian characterised by an exponent of 16. Currently used to emulate the
        'spike' in d14c data following the occurrence of an Miyake event
        Parameters
        ----------
        t : ndarray
            Scalar or vector input
        start_time : float
            Start time of a hypothetical Miyake event
        duration : float
            Duration of a hypothetical Miyake event
        area : float
            Total radiocarbon delivered by a hypothetical Miyake event (in production rate times years)
        Returns
        -------
        ndarray
            Super gaussian density
        """
        middle = start_time + duration / 2.
        height = area / duration
        return height * jnp.exp(- ((t - middle) / (1. / 1.93516 * duration)) ** 16.)

    @partial(jit, static_argnums=(0,))
    def simple_sinusoid(self, t, *args):
        """
        A simple sinusoid model for production rates in a period where a Miyake event is observed. Tunable parameters
        include,
        Start time: start time of the Miyake event
        Duration: duration of the Miyake event
        Phase: phase of the solar cycle during this period
        Area: total radiocarbon delivered by this Miyake event (in production rate times years)
        Parameters
        ----------
        t : ndarray
            Time values. Scalar or vector input
        args : ndarray | float
            Can be passed in as a ndarray or as individual floats. Must include start time, duration, phase and area
        Returns
        -------
        ndarray
            Production rate on t
        """
        start_time, duration, phase, area = jnp.array(list(args)).reshape(-1)
        height = self.super_gaussian(t, start_time, duration, area)
        production = self.steady_state_production + 0.18 * self.steady_state_production * jnp.sin(
            2 * np.pi / 11 * t + phase) + height
        return production

    @partial(jit, static_argnums=(0,))
    def flexible_sinusoid(self, t, *args):
        """
        A flexible sinusoid model for production rates in a period where a Miyake event is observed. Tunable parameters
        include:
        Start time: start time of the Miyake event
        Duration: duration of the Miyake event
        Phase: phase of the solar cycle during this period
        Area: total radiocarbon delivered by this Miyake event (in production rate times years)
        Amplitude: Amplitude of the solar cycle during this period
        Parameters
        ----------
        t : ndarray
            Time values. Scalar or vector input
        args : ndarray | float
            Can be passed in as a ndarray or as individual floats. Must include start time, duration, phase, area and
            amplitude
        Returns
        -------
        ndarray
            Production rate on t
        """
        start_time, duration, phase, area, amplitude = jnp.array(list(args)).reshape(-1)
        height = self.super_gaussian(t, start_time, duration, area)
        production = self.steady_state_production + amplitude * self.steady_state_production * jnp.sin(
            2 * np.pi / 11 * t + phase) + height
        return production

    @partial(jit, static_argnums=(0))
    def run_burnin(self, y0=None, params=()):
        """
        Calculates the C14 content of all the boxes within a carbon box model at the specified time values.
        Parameters
        ----------
        time_values : ndarray
            Time values
        y0 : ndarray, optional
            The initial contents of all boxes
        params : ndarray, optional
            Parameters for self.production
        Returns
        -------
        ndarray
            The value of each box in the carbon box at the specified time_values along with the steady state solution
            for the system
        """
        box_values, _ = self.cbm.run(self.burn_in_time, self.oversample, self.production, y0=y0, args=params)
        return box_values

    @partial(jit, static_argnums=(0))
    def run_event(self, y0=None, params=()):
        """
        Calculates the C14 content of all the boxes within a carbon box model at the specified time values.
        Parameters
        ----------
        time_values : ndarray
            Time values
        y0 : ndarray, optional
            The initial contents of all boxes
        params : ndarray, optional
            Parameters for self.production
        Returns
        -------
        ndarray
            The value of each box in the carbon box at the specified time_values along with the steady state solution
            for the system
        """
        box_values, _ = self.cbm.run(self.annual, self.oversample, self.production, y0=y0, args=params)
        return box_values

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
        burnin = self.run_burnin(y0=self.steady_state_y0, params=params)
        event = self.run_event(y0=burnin[-1, :], params=params)
        binned_data = self.cbm.bin_data(event[:, 1], self.oversample, self.annual, growth=self.growth)
        d14c = (binned_data - self.steady_state_y0[1]) / self.steady_state_y0[1] * 1000
        return d14c[self.mask] + self.offset

    @partial(jit, static_argnums=(0,))
    def dc14_fine(self, params=()):
        """
        Predict d14c on the same time sampling as self.time_data_fine.
        Parameters
        ----------
        params : ndarray, optional
            Parameters for self.production
        Returns
        -------
        ndarray
            Predicted d14c value
        """
        burnin = self.run_burnin(y0=self.steady_state_y0, params=params)
        event = self.run_event(y0=burnin[-1, :], params=params)
        d14c = (event[:, 1] - self.steady_state_y0[1]) / self.steady_state_y0[1] * 1000
        return d14c + self.offset

    @partial(jit, static_argnums=(0,))
    def log_prior_simple_sinusoid(self, params, low_bounds, high_bounds):
        """
        Computes the log prior likelihood of parameters of simple sinusoid model
        Parameters
        ----------
        params : ndarray
            Parameters of simple sinusoid model
        low_bounds : ndarray
            Lower bounds for 'params'
        high_bounds : ndarray
            Upper bounds for 'params'
        Returns
        -------
        float
            log prior likelihood
        """
        lp = 0
        lp += ((params[0] < low_bounds[0]) | (params[0] > high_bounds[0])) * -jnp.inf
        lp += ((params[1] < low_bounds[1]) | (params[1] > high_bounds[1])) * -jnp.inf
        lp += ((params[2] < low_bounds[2]) | (params[2] > high_bounds[2])) * -jnp.inf
        lp += ((params[3] < low_bounds[3]) | (params[3] > high_bounds[3])) * -jnp.inf
        return lp

    @partial(jit, static_argnums=(0,))
    def log_prior_flexible_sinusoid(self, params, low_bounds, high_bounds):
        """
        Computes the log prior likelihood of parameters of flexible sinusoid model
        Parameters
        ----------
        params : ndarray
            Parameters of flexible sinusoid model
        low_bounds : ndarray
            Lower bounds for 'params'
        high_bounds : ndarray
            Upper bounds for 'params'
        Returns
        -------
        float
            log prior likelihood
        """
        lp = 0
        lp += ((params[0] < low_bounds[0]) | (params[0] > high_bounds[0])) * -jnp.inf
        lp += ((params[1] < low_bounds[1]) | (params[1] > high_bounds[1])) * -jnp.inf
        lp += ((params[2] < low_bounds[2]) | (params[2] > high_bounds[2])) * -jnp.inf
        lp += ((params[3] < low_bounds[3]) | (params[3] > high_bounds[3])) * -jnp.inf
        lp += ((params[4] < low_bounds[4]) | (params[4] > high_bounds[4])) * -jnp.inf
        return lp

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, params=()):
        """
        Computes the gaussian log-likelihood of parameters of self.production
        Parameters
        ----------
        params : ndarray, optional
            Parameters of self.production
        Returns
        -------
        float
            Gaussian log-likelihood
        """
        d_14_c = self.dc14(params=params)
        chi2 = jnp.sum(((self.d14c_data - d_14_c) / self.d14c_data_error) ** 2)
        return -0.5 * chi2

    @partial(jit, static_argnums=(0,))
    def neg_log_likelihood(self, params=()):
        """
        Computes the negative gaussian log-likelihood of parameters of self.production
        Parameters
        ----------
        params : ndarray, optional
            Parameters of self.production
        Returns
        -------
        float
            Negative gaussian log-likelihood
        """
        return -1 * self.log_likelihood(params=params)

    @partial(jit, static_argnums=(0,))
    def log_joint_simple_sinusoid(self, params, low_bounds, high_bounds):
        """
        Computes the log joint likelihood of parameters of simple sinusoid model
        Parameters
        ----------
        params : ndarray
            Parameters of simple sinusoid model
        low_bounds : ndarray
            Lower bounds for 'params'
        high_bounds : ndarray
            Upper bounds for 'params'
        Returns
        -------
        float
            Log joint likelihood
        """
        lp = self.log_prior_simple_sinusoid(params, low_bounds, high_bounds)
        pos = self.log_likelihood(params=params)
        return lp + pos

    @partial(jit, static_argnums=(0,))
    def log_joint_flexible_sinusoid(self, params, low_bounds, high_bounds):
        """
        Computes the log joint likelihood of parameters of flexible sinusoid model
        Parameters
        ----------
        params : ndarray
            Parameters of flexible sinusoid model
        low_bounds : ndarray
            Lower bounds for 'params'
        high_bounds : ndarray
            Upper bounds for 'params'
        Returns
        -------
        float
            Log joint likelihood
        """
        lp = self.log_prior_flexible_sinusoid(params, low_bounds, high_bounds)
        pos = self.log_likelihood(params=params)
        return lp + pos

    @partial(jit, static_argnums=(0,))
    def neg_log_likelihood_gp(self, params):
        """
        Computes the negative log-likelihood of a set of control-points with respect to a Gaussian Process with
        constant mean and Matern-3/2 kernel.
        Parameters
        ----------
        params : ndarray
            An array of control-points. First control point is also the mean of the Gaussian Process
        Returns
        -------
        float
            Negative Gaussian Process log-likelihood
        """
        kernel = jax_terms.Matern32Term(sigma=2., rho=2.)
        gp = celerite2.jax.GaussianProcess(kernel, mean=params[0])
        gp.compute(self.control_points_time)
        return -gp.log_likelihood(params)

    @partial(jit, static_argnums=(0,))
    def log_likelihood_gp(self, params):
        """
        Computes the log-likelihood of a set of control-points with respect to a Gaussian Process with
        constant mean and Matern-3/2 kernel.
        Parameters
        ----------
        params : ndarray
            An array of control-points. First control point is also the mean of the Gaussian Process
        Returns
        -------
        float
            Gaussian Process log-likelihood
        """
        control_points = params
        mean = params[0]
        kernel = jax_terms.Matern32Term(sigma=2., rho=2.)
        gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
        gp.compute(self.control_points_time)
        return gp.log_likelihood(control_points)

    @partial(jit, static_argnums=(0,))
    def log_joint_gp(self, params=()):
        """
        Computes the log joint likelihood of control-points for non-parametric inferences. Currently used as the
        likelihood function for Monte Carlo sampling.
        Parameters
        ----------
        params : ndarray
            An array of control-points. First control point is also the mean of the Gaussian Process
        Returns
        -------
        float
            Log joint likelihood
        """
        return self.log_likelihood(params=params) + self.log_likelihood_gp(params)

    @partial(jit, static_argnums=(0,))
    def neg_log_joint_gp(self, params=()):
        """
        Computes the negative log joint likelihood of control-points for non-parametric inferences. Currently used as
        the objective function for fitting the set of control-points via numerical optimization.
        Parameters
        ----------
        params : ndarray
            An array of control-points. First control point is also the mean of the Gaussian Process
        Returns
        -------
        float
            Negative log joint likelihood
        """
        return self.neg_log_likelihood(params=params) + self.neg_log_likelihood_gp(params)

    @partial(jit, static_argnums=(0,))
    def neg_grad_log_joint_gp(self, params=()):
        """
        Computes the negative gradient of the log joint likelihood of control-points.
        Parameters
        ----------
        params : ndarray
            An array of control-points. First control point is also the mean of the Gaussian Process
        Returns
        -------
        float
            Negative gradient of log joint likelihood
        """
        return grad(self.neg_log_joint_gp)(params)

    def fit_ControlPoints(self, low_bound=0):
        """
        Fits the control-points by minimizing the negative log joint likelihood.
        Parameters
        ----------
        low_bound : int, optional
            The minimum value each control-point can take. 0 by default.
        Returns
        -------
        OptimizeResult
            Scipy OptimizeResult object
        """
        initial = self.steady_state_production * jnp.ones((len(self.control_points_time),))
        bounds = tuple([(low_bound, None)] * len(initial))

        if self.gp:
            soln = scipy.optimize.minimize(self.neg_log_joint_gp, initial, bounds=bounds,
                                           options={'maxiter': 20000})
        else:
            soln = scipy.optimize.minimize(self.neg_log_likelihood, initial, bounds=bounds,
                                           method="L-BFGS-B", options={'maxiter': 20000})
        return soln

    def plot_recovery(self, chain, time_data=None, true_production=None, size=100, alpha=0.2):
        """
        Takes a chain of Markov Chain Monte Carlo walks and the true production rates, plots the predicted production
        rate from different samples of the chain against the true production rates.
        Parameters
        ----------
        chain : ndarray
            A chain of MCMC walks
        time_data : ndarray, optional
            Array of time sampling on which production rates will be evaluated
        true_production : ndarray, optional
            True production rates on 'time_data'
        size : int, optional
            The number of samples randomly chosen from 'chain'
        alpha : float, optional
            Parameter for blending, between 0-1.
        Returns
        -------
        figure
            plot of samples against true production rates
        """
        mean = np.mean(chain, axis=0)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10), sharex=True)
        top_n = np.random.permutation(len(chain))[:size]
        ax1.errorbar(self.time_data, self.d14c_data, yerr=self.d14c_data_error,
                     fmt="o", color="k", fillstyle="full", capsize=3, markersize=4, label="noisy d14c")
        for i in tqdm(top_n):
            d14c = self.dc14_fine(params=chain[i, :])
            ax1.plot(self.time_data_fine, d14c, color="g", alpha=alpha)
            control_points_fine = self.production(self.time_data_fine, (chain[i, :],))
            ax2.plot(self.time_data_fine, control_points_fine, color="g", alpha=alpha)
        control_points_fine = self.production(self.time_data_fine, (mean,))
        ax2.plot(self.time_data_fine, control_points_fine, "r", label="sample mean production rate")
        ax1.set_ylabel("$\Delta^{14}$C (‰)");
        ax2.set_ylabel("Production rate ($cm^2s^{-1}$)")
        ax2.set_xlabel("Calendar Year")
        if (true_production is not None) & (time_data is not None):
            ax2.plot(time_data, true_production, 'k', label="true production rate")
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fancybox=True);
        ax1.legend();

    def plot_samples(self, chain, nwalkers, size=100, alpha=0.2):
        """
        Takes a chain of Markov Chain Monte Carlo walks and plots the predicted production rate from different samples
        of the chain.
        Parameters
        ----------
        chain : ndarray
            A chain of MCMC walks
        nwalkers : int
            Number of walkers of 'chain'
        size : int, optional
            The number of samples randomly chosen from 'chain'
        alpha : float, optional
            Parameter for blending, between 0-1.
        Returns
        -------
        figure
            plot of samples
        """
        c = ChainConsumer().add_chain(chain, walkers=nwalkers)
        mle = []
        for lst in c.analysis.get_summary().values():
            mle.append(lst[1])

        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        for s in chain[np.random.randint(len(chain), size=size)]:
            d_c_14_fine = self.dc14_fine(params=s)
            ax1.plot(self.time_data_fine, d_c_14_fine, alpha=alpha, color="g")

        d_c_14_coarse = self.dc14(params=mle)
        d_c_14_fine = self.dc14_fine(params=mle)
        ax1.plot(self.time_data_fine, d_c_14_fine, color="k")

        ax1.plot(self.time_data, d_c_14_coarse, "o", color="k", fillstyle="none", markersize=7,
                 label="fitted $\Delta^{14}$C")
        ax1.errorbar(self.time_data, self.d14c_data,
                     yerr=self.d14c_data_error, fmt="o", color="k", fillstyle="full", capsize=3, markersize=7,
                     label="$\Delta^{14}$C data")
        ax1.set_ylabel("$\Delta^{14}$C (‰)")
        ax1.legend(loc="lower right")
        fig.subplots_adjust(hspace=0.05)

        for s in chain[np.random.randint(len(chain), size=10)]:
            production_rate = self.production(self.time_data_fine, *s)
            ax2.plot(self.time_data_fine, production_rate, alpha=0.25, color="g")

        production_rate = self.production(self.time_data_fine, *mle)
        ax2.plot(self.time_data_fine, production_rate, color="k", lw=2, label="MLE")
        ax2.set_ylim(jnp.min(production_rate) * 0.8, jnp.max(production_rate) * 1.1);
        ax2.set_xlabel("Calendar Year (CE)");
        ax2.set_ylabel("Production rate ($cm^2s^{-1}$)");
        ax2.legend(loc="upper left")

    # @partial(jit, static_argnums=(0,))
    # def sum_interp_gp(self, *args):
    #     mu = self.interp_gp(self.annual, *args)
    #     return jnp.sum(mu)
    #
    # @partial(jit, static_argnums=(0,))
    # def grad_sum_interp_gp(self, *args):
    #     return grad(self.sum_interp_gp)(*args)

class MultiFitter(CarbonFitter):
    """
    A class for parametric inference of d14c data from a common time period using an ensemble of SingleFitter.
    Does parameter fitting, likelihood evaluations, Monte Carlo sampling, plotting and more.
    """
    def __init__(self, sf=None):
        """
        Initializes a MultiFitter object. If sf is not None it should be a list of SingleFitter objects.
        Parameters
        ----------
        params : ndarray
            Parameters of a parametric model
        Returns
        -------
        float
            Log-likelihood
        """
        # TODO: add start, end filter
        if isinstance(sf, list):
            valid_sf = True
            for object in sf:
                valid_sf = valid_sf and isinstance(object, SingleFitter)
            if valid_sf:
                self.MultiFitter = sf
            else:
                raise ValueError("sf should be a list of SingleFitter objects")
        elif sf is None:
            self.MultiFitter = []
            self.idx = 0
            self.start = None
            self.end = None
        else:
            raise ValueError("Invalid sf. sf should be None or a list of SingleFitter objects")

    def compile(self):
        for sf in self.MultiFitter:
            sf.multi_mask = jnp.in1d(self.annual, sf.time_data)

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
        sf = self.MultiFitter[self.idx]
        burnin = sf.run_burnin(y0=sf.steady_state_y0, params=params)
        event = sf.run_event(y0=burnin[-1, :], params=params)
        binned_data = sf.cbm.bin_data(event[:, 1], sf.oversample, sf.annual, growth=sf.growth)
        d14c = (binned_data - sf.steady_state_y0[1]) / sf.steady_state_y0[1] * 1000
        return d14c

    @partial(jit, static_argnums=(0, 1))
    def log_likelihood(self, sf, d14c):
        """
        Computes the gaussian log-likelihood of parameters of self.production
        Parameters
        ----------
        params : ndarray, optional
            Parameters of self.production
        Returns
        -------
        float
            Gaussian log-likelihood
        """
        d14c = d14c[sf.multi_mask] + sf.offset
        chi2 = jnp.sum(((sf.d14c_data - d14c) / sf.d14c_data_error) ** 2)
        return -0.5 * chi2

    @partial(jit, static_argnums=(0,))
    def multi_likelihood(self, params):
        """
        Computes the ensemble log-likelihood of parameters of some parametric model, across multiple d14c datasets
        Parameters
        ----------
        params : ndarray
            Parameters of a parametric model
        Returns
        -------
        float
            Log-likelihood
        """
        like = 0
        d14c = self.dc14(params)
        for sf in self.MultiFitter:
            like += self.log_likelihood(sf, d14c)
        return like

    @partial(jit, static_argnums=(0,))
    def log_prior_simple_sinusoid(self, params, low_bounds, high_bounds):
        """
        Computes the log prior likelihood of parameters of simple sinusoid model
        Parameters
        ----------
        params : ndarray
            Parameters of simple sinusoid model
        low_bounds : ndarray
            Lower bounds for 'params'
        high_bounds : ndarray
            Upper bounds for 'params'
        Returns
        -------
        float
            log prior likelihood
        """
        lp = 0
        lp += ((params[0] < low_bounds[0]) | (params[0] > high_bounds[0])) * -jnp.inf
        lp += ((params[1] < low_bounds[1]) | (params[1] > high_bounds[1])) * -jnp.inf
        lp += ((params[2] < low_bounds[2]) | (params[2] > high_bounds[2])) * -jnp.inf
        lp += ((params[3] < low_bounds[3]) | (params[3] > high_bounds[3])) * -jnp.inf
        return lp

    @partial(jit, static_argnums=(0,))
    def log_prior_flexible_sinusoid(self, params, low_bounds, high_bounds):
        """
        Computes the log prior likelihood of parameters of flexible sinusoid model
        Parameters
        ----------
        params : ndarray
            Parameters of flexible sinusoid model
        low_bounds : ndarray
            Lower bounds for 'params'
        high_bounds : ndarray
            Upper bounds for 'params'
        Returns
        -------
        float
            log prior likelihood
        """
        lp = 0
        lp += ((params[0] < low_bounds[0]) | (params[0] > high_bounds[0])) * -jnp.inf
        lp += ((params[1] < low_bounds[1]) | (params[1] > high_bounds[1])) * -jnp.inf
        lp += ((params[2] < low_bounds[2]) | (params[2] > high_bounds[2])) * -jnp.inf
        lp += ((params[3] < low_bounds[3]) | (params[3] > high_bounds[3])) * -jnp.inf
        lp += ((params[4] < low_bounds[4]) | (params[4] > high_bounds[4])) * -jnp.inf
        return lp

    @partial(jit, static_argnums=(0,))
    def log_joint_simple_sinusoid(self, params, low_bounds, high_bounds):
        """
        Computes the log joint likelihood of parameters of simple sinusoid model
        Parameters
        ----------
        params : ndarray
            Parameters of simple sinusoid model
        low_bounds : ndarray
            Lower bounds for 'params'
        high_bounds : ndarray
            Upper bounds for 'params'
        Returns
        -------
        float
            Log joint likelihood
        """
        lp = self.log_prior_simple_sinusoid(params, low_bounds, high_bounds)
        pos = self.multi_likelihood(params=params)
        return lp + pos

    @partial(jit, static_argnums=(0,))
    def log_joint_flexible_sinusoid(self, params, low_bounds, high_bounds):
        """
        Computes the log joint likelihood of parameters of flexible sinusoid model
        Parameters
        ----------
        params : ndarray
            Parameters of flexible sinusoid model
        low_bounds : ndarray
            Lower bounds for 'params'
        high_bounds : ndarray
            Upper bounds for 'params'
        Returns
        -------
        float
            Log joint likelihood
        """
        lp = self.log_prior_flexible_sinusoid(params, low_bounds, high_bounds)
        pos = self.multi_likelihood(params=params)
        return lp + pos

    def add_SingleFitter(self, sf):
        """
        Adds a SingleFitter Object to a Multifitter Object
        Parameters
        ----------
        sf : SingleFitter
            SingleFitter Object
        Returns
        -------
        """
        if not self.start:
            self.start = sf.start
            self.end = sf.end
            self.annual = jnp.arange(self.start, self.end + 1)
        elif self.start != sf.start:
            raise ValueError("start date for SingleFitters must be consistent. Got {}, expected {}".format(sf.start,
                             self.start))
        if self.end < sf.end:
            self.idx = len(self.MultiFitter)
            self.end = sf.end
            self.annual = jnp.arange(self.start, self.end + 1)
        self.MultiFitter.append(sf)

    def get_time_period(self):
        """
        Retrieves the earliest and the latest time sampling covered by the SingleFitters
        Parameters
        ----------
        Returns
        -------
        tuple
            the start and the end of the time sampling
        """
        start = jnp.min(jnp.array([sf.start for sf in self.MultiFitter]))
        end = jnp.max(jnp.array([sf.end for sf in self.MultiFitter]))
        return start, end

def get_data(path=None, event=None, hemisphere='north'):
    """
    Retrieves the earliest and the latest time sampling covered by the SingleFitters.
    Parameters
    ----------
    path : str, optional
        When specified it is the relative path to the directory where the data is stored. Only one of path and event
        should be specified at any time.
    event : str, optional
        Identifier of known Miyake events. When specified it takes values from: '660BCE', '775AD', '993AD', '5259BCE',
        '5410BCE', '7176BCE'.
    hemisphere : str, optional
        hemispheric parameter for Carbon Box Model. Used to retrieve the correct data files when event is specified.
    Returns
    -------
    list
        A list of file names to be loaded into SingleFitter
    """
    if path:
        file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    elif event in ['660BCE', '775AD', '993AD', '5259BCE', '5410BCE', '7176BCE']:
        if hemisphere == 'north':
            file = 'data/datasets/' + event + '/NH'
        elif hemisphere == 'south':
            file = 'data/datasets/' + event + '/SH'
        else:
            raise ValueError("Invalid hemisphere, hemisphere must be from: 'north', 'south'")
        path = os.path.join(os.path.dirname(__file__), file)
        file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        raise ValueError("Invalid path, or event is not from: '660BCE', '775AD', '993AD', '5259BCE', '5410BCE', '7176BCE'")
    return file_names

def sample_event(year, mf, sampler='MCMC', production_model='simple_sinusoid', burnin=500, production=1000,
                 params=(), low_bounds=None, high_bounds=None):
    """
    Runs Monte Carlo sampler on a Miyake event.
    Parameters
    ----------
    year : float
        The calender year which the Miyake event is supposed to have occurred.
    mf : MultiFitter
        MultiFitter object that enables likelihood function evaluations.
    sampler : str, optional
        Monte Carlo sampler. 'MCMC' for Markov Chain Monte Carlo, 'NS' for Nested Sampling. 'MCMC' by default.
    production_model : str | callable, optional
        The d14c production rate model of the SingleFitters. 'simple_sinusoid' by default.
    burnin : int, optional
        Number of burn-in steps for Markov Chain Monte Carlo, 500 by default.
    production : int, optional
        Number of production steps for Markov Chain Monte Carlo, 1000 by default.
    params : ndarray, optional
        Initial parameters for Monte Carlo samplers. Required when custom production rate model is used.
    low_bounds : ndarray, optional
        Lower bound for parameters. Required when custom production rate model is used.
    high_bounds : ndarray, optional
        Upper bound for parameters. Required when custom production rate model is used.
    Returns
    -------
    result
        MCMC sampler or NS sampler
    """
    start, end = (mf.start, mf.end)
    if sampler == 'MCMC':
        if production_model == 'simple_sinusoid':
            default_params = np.array([year, 1./12, np.pi/2., 81./12])
            result = mf.MarkovChainSampler(default_params,
                                            likelihood = mf.log_joint_simple_sinusoid,
                                            burnin = burnin,
                                            production = production,
                                            args = (jnp.array([year-5, 0., -jnp.pi, 0.]),
                                            jnp.array([year+5, 5., jnp.pi, 15.]))
                                           )
        elif production_model == 'flexible_sinusoid':
            default_params = np.array([year, 1./12, np.pi/2., 81./12, 0.18])
            result = mf.MarkovChainSampler(default_params,
                                            likelihood = mf.log_joint_flexible_sinusoid,
                                            burnin = burnin,
                                            production = production,
                                            args = (jnp.array([year-5, 0., -jnp.pi, 0., 0.]),
                                            jnp.array([year+5, 5., jnp.pi, 15., 2.]))
                                           )
        elif callable(production_model):
            def log_joint_likelihood(params, low_bounds, high_bounds):
                lp = 0
                lp += jnp.all(
                    (params < low_bounds) | (params > high_bounds)
                ) * -jnp.inf
                pos = mf.multi_likelihood(params=params)
                return lp + pos
            result = mf.MarkovChainSampler(params,
                                            likelihood = log_joint_likelihood,
                                            burnin = burnin,
                                            production = production,
                                           args = (low_bounds, high_bounds)
                                           )
        else:
            raise ValueError("Invalid production model")
    elif sampler == 'NS':
        print("Running Nested Sampling...")
        if production_model == 'simple_sinusoid':
            default_params = np.array([year, 1. / 12, np.pi / 2., 81. / 12])
            result = mf.NestedSampler(default_params,
                                      likelihood=mf.multi_likelihood,
                                      low_bound=jnp.array([start, 0., -jnp.pi, 0.]),
                                      high_bound=jnp.array([end, 5., jnp.pi, 15.])
                                      )
        elif production_model == 'flexible_sinusoid':
            default_params = np.array([year, 1. / 12, np.pi / 2., 81. / 12, 0.18])
            result = mf.NestedSampler(default_params,
                                      likelihood=mf.multi_likelihood,
                                      low_bound=jnp.array([start, 0., -jnp.pi, 0., 0.]),
                                      high_bound=jnp.array([end, 5., jnp.pi, 15., 2.])
                                      )
        elif callable(production_model):
            result = mf.NestedSampler(params,
                                      likelihood=mf.multi_likelihood,
                                      low_bound=low_bounds,
                                      high_bound=high_bounds
                                      )
        else:
            raise ValueError("Invalid production model")
        print("Done")
    else:
        raise ValueError("Invalid sampler type. sampler must be one of the following: MCMC, NS")
    return result

def fit_event(year, event=None, path=None, production_model='simple_sinusoid', cbm_model='Guttler14', box='Troposphere',
              hemisphere='north', sampler=None, burnin=500, production=1000, params=(), low_bounds=None,
              high_bounds=None, mf=None, oversample=108):
    """
    Fits a Miyake event.
    Parameters
    ----------
    year : float
        The calender year which the Miyake event is supposed to have occurred.
    event : str, optional
        Identifier of known Miyake events. When specified it takes values from: '660BCE', '775AD', '993AD', '5259BCE',
        '5410BCE', '7176BCE'.
    path : str, optional
        When specified it is the relative path to the directory where the data is stored. Only one of path and event
        should be specified at any time.
    production_model : str | callable, optional
        The d14c production rate model of the SingleFitters. 'simple_sinusoid' by default.
    cbm_model : str, optional
        Name of a Carbon Box Model. Must be one from: Miyake17, Brehm21, Guttler14, Buntgen18.
    box : str, optional
        The specific box at which to calculate the d14c. 'Troposphere' by default
    hemisphere : str, optional
        hemispheric parameter for Carbon Box Model. Used to retrieve the correct data files when event is specified.
    sampler : str, optional
        Monte Carlo sampler. 'MCMC' for Markov Chain Monte Carlo, 'NS' for Nested Sampling. 'MCMC' by default.
    burnin : int, optional
        Number of burn-in steps for Markov Chain Monte Carlo, 500 by default.
    production : int, optional
        Number of production steps for Markov Chain Monte Carlo, 1000 by default.
    params : ndarray, optional
        Initial parameters for Monte Carlo samplers. Required when custom production rate model is used.
    low_bounds : ndarray, optional
        Lower bound for parameters. Required when custom production rate model is used.
    high_bounds : ndarray, optional
        Upper bound for parameters. Required when custom production rate model is used.
    mf : MultiFitter
        MultiFitter object that enables likelihood function evaluations. If None, a new MultiFitter Object will be
        initialized
    Returns
    -------
    mf : MultiFitter
        MultiFitter object that enables likelihood function evaluations.
    result : ndrray | NestedSampler Object
        MCMC sampler or NS sampler
    """
    if not mf:
        mf = MultiFitter()
    cbm = ticktack.load_presaved_model(cbm_model, production_rate_units='atoms/cm^2/s')
    if event:
        file_names = get_data(event=event, hemisphere=hemisphere)
        print("Retrieving data...")
        for file in tqdm(file_names):
            if hemisphere == 'north':
                file_name = 'data/datasets/' + event + '/NH/' + file
            else:
                file_name = 'data/datasets/' + event + '/SH/' + file
            sf = SingleFitter(cbm, box=box, hemisphere=hemisphere)
            sf.load_data(os.path.join(os.path.dirname(__file__), file_name), oversample=oversample)
            sf.prepare_function(model=production_model)
            mf.add_SingleFitter(sf)
    elif path:
        file_names = get_data(path=path)
        print("Retrieving data...")
        for file_name in tqdm(file_names):
            sf = SingleFitter(cbm, box=box, hemisphere=hemisphere)
            sf.load_data(path + '/' + file_name, oversample=oversample)
            sf.prepare_function(model=production_model)
            mf.add_SingleFitter(sf)
    mf.compile()
    if not sampler:
        return mf
    else:
        return mf, sample_event(year, mf, sampler, params=params, burnin=burnin, production=production,
                                production_model=production_model, low_bounds=low_bounds, high_bounds=high_bounds)
