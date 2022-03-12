import jax
import h5py
import functools


@jax.jit
def convert_production_rate(production_rate, units):
    """
    Convert the production rate from units to "kg/yr"

    Parameters
    ----------
    production_rate : jax.numpy.float64
        The production rate at an instant in time
    units : jax.DeviceArray
        Corresponds to [1, 0] if the units are "atoms/cm^2/s"
        else [0, 1]

    Returns
    -------
    jax.numpy.float64 
        The production rate at the current time in "kg/yr"
    """
    return jax.lax.cond(units, 
            lambda: production_rate * 14.003242 / 6.022 * 
                5.11 * 31536. / 1.e5,
            lambda: production_rate)


@functools.partial(jax.jit, static_argnums=(6))
def derivative(y, t, args, matrix=None, steady_state=None, 
        projection=None, production=None):
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
    production_rate = production(t, *args) - steady_state
    production_rate = convert_production_rate(production_rate, True)
    return state + production_rate * projection


@functools.partial(jax.jit, static_argnums=(7))
def run(time, y0, equilibrium, args, matrix, steady_state, projection,
        production):
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
        return derivative(y, t, args, matrix=matrix, 
                steady_state=steady_state, projection=projection,
                production=production)

    states = jax.experimental.ode.odeint(dydt, y0 - equilibrium,
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


@jax.jit 
def equilibrate_brehm(production_rate, matrix, projection, units):
    """
    Uses a production rate to determine the box concentrations 
    if that production is the steady state production.

    Parameters
    ----------
    production_rate : jax.numpy.float64
        A production rate in any acceptible units.
    matrix : jax.DeviceArray
        The transfer matrix of the system
    projection : jax.DeviceArray
        The projection of the production into the atmosphere

    Returns
    -------
    DeviceArray
        The steady state box values
    """
    production_rate = convert_production_rate(production_rate, units)
    production_rate = - production_rate * projection
    return jax.numpy.linalg.solve(matrix, production_rate)


@jax.jit
def build_matrix(fluxes, contents, decay_rate):
    """
    Constructs a transfer matrix for a model. 

    Parameters
    ----------
    fluxes : jax.DeviceArray
        The absolute fluxes of the model. Should be a square matrix
    contents : jax.DeviceArray
        The reservoir contents. Should be a vector
    decay_rate : jax.numpy.float64
        The rate of carbon decay

    Returns
    -------
    DeviceArray
        The transfer matrix of the system
    """
    contents = contents.reshape(-1, 1)
    decay_matrix = jax.numpy.diag(
            jax.numpy.array([decay_rate] * contents.size))
    relative_fluxes = fluxes / contents
    box_retention = jax.numpy.diag(jax.numpy.sum(relative_fluxes, axis=1))
    return relative_fluxes.T - box_retention - decay_matrix


def load(file_name, /, production_rate_units="atoms/cm^2/s", 
        flow_rate_units="Gt/yr"):
    """
    Takes the model saved in file_name which is assumed to have 
    the .hd5 extension.
    
    Parameters
    ----------
    file_name : str
        The address of the saved model file with a .hd5 extension
    production_rate_units : str
        The units of the production rate.
    flow_rate_units : str
        The units of the flows 
    """
    model = h5py.File(file_name, "r")
    
    box_model = CarbonBoxModel(production_rate_units, flow_rate_units)
    box_model._production_coefficients = jax.numpy.array(
            model["production coefficients"])
    box_model._reservoir_content = jax.numpy.array(
            model["reservoir content"])
    box_model._matrix = build_matrix(
            jax.numpy.array(model["fluxes"]), 
            jax.numpy.array(model["reservoir content"]), 
            box_model.DECAY_RATE) 
    return box_model


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
        self._production_coefficients = None # Vector of dimensions
        self._reservoir_content = None # Vector of dimensions
        self._equilibrium = None # Vector of dimensions
        self._steady_state_production = None
    

    def equilibrate(self, production_rate=1.88):
        """
        Equilibrate the model to some production rate assumed to be the 
        steady state production.

        Parameters
        ----------
        production_rate : jax.numpy.float64
            The seady state production
        """
        if self._production_rate_units == "atoms/cm^2/s":
            conversion_array = True
        else:
            conversion_array = False
        
        self._steady_state_production = production_rate
        self._equilibrium = equilibrate_brehm(production_rate, 
                self._matrix, self._production_coefficients, 
                conversion_array)


    @functools.partial(jax.jit, static_argnums=(0, 1))
    def run(self, production, time, y0, args=()):
        """
        Runs the model saving it at the values in time.

        Parameters
        ----------
        production : Function
            The production model evaluated at time with args
        time : jax.DeviceArray
            The time values at which to save the box values
        args : Tuple
            The arguments to the production model

        Return
        jax.DeviceArray
            The contents of the boxes evaluated at time
        """
        return run(time, y0 , equilibrium=self._equilibrium, 
                args=args, matrix=self._matrix, 
                steady_state=self._steady_state_production, 
                projection=self._production_coefficients, 
                production=production)
        
        

