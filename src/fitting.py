import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import rcParams
import celerite
from celerite import terms
from celerite.modeling import Model
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(1,'../src')

import re
import jax.numpy as jnp
from jax import grad, jit, partial
import ticktack

rcParams['figure.figsize'] = (16.0, 8.0)