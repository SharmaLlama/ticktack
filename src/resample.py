import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from functools import partial
from jax import jit
import jax.numpy as jnp
import warnings
warnings.filterwarnings("ignore")


class Resampler:

    def load_data(self, time_data, d14c_data, d14c_data_error):
        self.time_data = time_data
        self.d14c_data = d14c_data
        self.d14c_data_error = d14c_data_error
        self.start = np.nanmin(self.time_data)
        self.end = np.nanmax(self.time_data)
        self.annual = jnp.arange(self.start, self.end + 1)

    def super_gaussian(self, x, sz, order=30):
        return np.exp(-(x / sz) ** order)

    def bandpass_super_gaussian(self, signal, l, sz, order=30):
        if np.ndim(signal) == 1:
            signal = signal.reshape(1, -1)
        n = signal.shape[1]
        freq = fftfreq(n, d=1)
        mask = self.super_gaussian(np.abs(freq) - l, sz, order=order).reshape(1, -1)
        new_signal = ifft(fft(signal) * mask)
        if new_signal.shape[0] == 1:
            return np.squeeze(new_signal)
        return new_signal

    def signal_resample(self, size=10000, l=1./9.677, sz=0.05, order=30):
        samples = np.random.multivariate_normal(self.d14c_data, np.diag(self.d14c_data_error ** 2), size=size)
        f = interp1d(self.time_data, samples, kind="cubic")
        interpolation = f(self.annual)
        new_sig = self.bandpass_super_gaussian(interpolation, l, sz, order=order)
        return new_sig

