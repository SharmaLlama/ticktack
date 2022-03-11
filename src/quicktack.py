import jax

PRODUCTION_CONVERTER = jax.numpy.array(
    [[14.003242 / 6.022 * 5.11 * 31536. / 1.e5, 0.0],
        [0.0, 1.0]]
)


@jax.jit
def convert_production_rate(production_rate, units, /, 
        conversion=PRODUCTION_CONVERTER):
    """
    Convert the production rate from units to "kg/yr"

    Parameters
    ----------
    production_rate : jax.numpy.float64
        The production rate at an instant in time
    units : jax.DeviceArray

    Returns
    -------
    jax.numpy.float64 
        The production rate at the current time in "kg/yr"
    """
    return production * jax.numpy.sum(conversion @ units)


@jax.jit
def derivative(y, t, *args, /, production=None, matrix=None, 
        steady_state=None, projection=None):
    """
    The derivative of the carbon box model at time t and state y

    Parameters
    ----------
    y : jax.DeviceArray
        The state at the previous increment
    t : jax.numpy.float64
        The current time
    *args: Tuple
        The arguments to the production function
    production: CompiledFunction
        The production function model
    matrix: jax.DeviceArray
        A square transfer matrix for the model
    steady_state: jax.numpy.float64
        The steady state production of the model
    projection: jax.DeviceArray
        The projection of the production model into the atmosphere
    units: CompiledFunction
        A function to convert the production rate into the correct units

    Returns
    -------
    jax.DeviceArray
        The state at the new time t
    """
    state = jax.numpy.matmul(matrix, y)
    production = production(t, *args) - steady_state
    production = convert_production_rate(production)
    return state + production * projection


@jax.jit
def run(derivative, time, y0,/, equilibrium=None, production=None, args=(),
        matrix=None, steady_state=None, projection=None):
    """
    Calculates the box values over the time series that is passed. 
    
    Parameters
    ----------
    derivative : CompiledFunction
        The integratable rate of change of the carbon box model.
    time : jax.DeviceArray
        The time values to evaluate the reservoir contents.
    production : CompiledFunction
        The production model.
    args : Tuple
        The arguments of the production model.

    Returns
    -------
    DeviceArray
        The reservoir contents evaluated across the time array.
    """
    def dydt(y, t): 
        return derivative(y, t, *args, production=production,
                matrix=matrix, steady_state=steady_state, 
                projection=projection)

    states = jax.experimental.ode.odeint(derivative, y0 - equlibrium,
            time, atol=1e-15, rtol=1e-15) + equilibrium

    return states, equilibrium

    
@jax.jit
def bin_data():

class CarbonBoxModel:
    """
    Template for an atmospheric model that uses a system of differential
    equations to represent the carbon cycle. The atmosphere is divided 
    into logical reservoirs with annual fluxes between them. For more 
    information on carbon box models some check out the ANU simple carbon 
    project:

        https://openresearch-repository.anu.edu.au/handle/1885/203769

    """
    DECAY_RATE = jax.numpy.log(2) / 5700 # Decay of C14

    def __init__(self, production_rate_units="kg/yr", 
            flow_rate_units="Gt/yr"):
        """
        Construct an empty template for a carbon box model.
        Parameters
        ----------
        production_rate_units : str, optional
            Units of the production model, either "atoms/cm^2/s" or "kg/yr"
        flow_rate_units : str, optional
            Units for the flow rate, either "Gt/yr" or "1/yr".
        """
        self._production_rate_units = production_rate_units
        self._flow_rate_units = flow_rate_units
        self._matrix = None # Square matrix of model dimensions 
        self._n_nodes = None # Dimensions of the model 
        self._production_coefficients = None # Vector of dimensions
        self._fluxes = None # Weighted adjacency matrix 
        self._reservoir_content = None # Vector of dimensions
        self._equilibrium = None # Vector of dimensions
    

