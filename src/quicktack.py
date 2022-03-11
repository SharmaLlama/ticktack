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
    
    def use_self(self, method):
        return method

