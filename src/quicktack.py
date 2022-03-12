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
def bin_data(start, end, dc14, time):
    """
    Takes the average of each year between start and end. This becomes
    the value for the year.

    Parameters
    ----------
    start : jax.numpy.float64
        A decimal representation of the point in the year to strart 
        collecting the simulated dc14
    end : jax.numpy.float64
        A decimal representation of the end point in the year to 
        stop collection values
    data : jax.DeviceArray(2 by n)
        A two dimensional array containing time and dc14 simulation
        data. The array is oriented in row major format for like 
        values

    Returns:
    DeviceArray
        The average dc14 in each year.
    """
    @jax.vmap
    def in_year(year, /, time=time):
        return (year <= time) & (time < (year + 1))

    decimal_time = time - jax.numpy.floor(time)
    use_values = (start < decimal_time) & (decimal_time < end)
    years = jax.numpy.arange(time.max().floor(), time.min().floor())
    year_mask = in_year(years)
    year_value_mask = year_mask * years
    
    # The rows of the matrix that is constructed above represent 
    # a mask for each year 
    # 
    # 1, 1, 1, 0, 0, 0  First year 
    # 0, 0, 0, 1, 1, 0  Second year
    # 0, 0, 0, 0, 0, 1  Third year
    # 
    # The matrix multiplication the produces a vector with the same
    # dimesnions as the data that contains the sum of the values 
    # isolated by each row.
    # 
    # 1, 1, 1, 0, 0, 0 * -14.5 = -38.2
    # 0, 0, 0, 1, 1, 0   -12.5   -15.6
    # 0, 0, 0, 0, 0, 1   -11.2   -6.6
    #                    -10.2
    #                    -5.4
    #                    -1.2
    # 
    # Taking the sum along the year we can then compute the average 
    # By dividing across elementwise.

    sum_in_year = year_value_maks @ dc14
    points_in_year = jax.numpy.sum(year_value_mask, axis=0)
    return sum_in_year / points_in_year






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
    

