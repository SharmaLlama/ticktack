# ticktack
A lightweight, open-source carbon box modelling library, adapted for modelling tree ring radiocarbon time series.

## Contributors

[Utkarsh Sharma](https://github.com/SharmaLlama), [Qingyuan Zhang](https://github.com/qingyuanzhang3), [Benjamin Pope](https://github.com/benjaminpope)

## Overview

Radiocarbon measurements from tree rings allow us to recover measurements of cosmic radiation from the distant past, and exquisitely calibrate carbon dating of archaeological sites. But in order to infer cosmic production rates from raw ΔC14 data, we need to model the entire global carbon cycle, from the production of radiocarbon in the stratosphere and troposphere to its uptake by the oceans and biosphere. Many such competing models exist, in which the Earth system is partitioned into 'boxes' with reservoirs of C12, C14, and coefficients of flow between them.

`ticktack` is the first open-source package for radiocarbon box modelling, allowing you to specify your own model or load a model with the same parameters as several leading closed-source models. Built in Python on [Google Jax](https://github.com/google/jax), it solves the carbon box ordinary differential equations using the Runge-Kutta method, on arbitrarily fine time grids and with arbitrary production rates. This forwards model is connected via a simple API to Bayesian inference using MCMC: currently we support only [emcee](https://emcee.readthedocs.io/), but implementations are in progress of HMC and nested sampling.  

## Installation

The easiest way to install is from PyPI: just use

`pip install ticktack`

To install from source: clone [this git repo](https://github.com/SharmaLlama/ticktack), enter the directory, and run

`python setup.py install`

## Running the Tests

To run the tests just, install [pytest](https://docs.pytest.org/en/7.1.x/contents.html), navigate to the `tests/` directory, and type `pytest` at the command line. `pytest` will automatically detect the tests.

## License

We invite anyone interested to use and modify this code under a MIT license.

## Name

The 'little boxes' in [Malvina Reynolds' famous song](https://www.youtube.com/watch?v=2_2lGkEU4Xs) are all made of ticky-tacky, and they all look just the same. Here we provide an open-source toolkit for reproducing and extending carbon box models for radiocarbon analysis, and we expect they will be as interchangeable as Malvina Reynolds' boxes!
