import h5py
import hdfdict
import jax.numpy as jnp
import scipy as scipy
import scipy.integrate
import scipy.optimize
from jax import jit
import jax
from functools import partial
from jax.config import config
import pkg_resources
from typing import Union
from jax.lax import cond, dynamic_update_slice, fori_loop, dynamic_slice
from jax.experimental.ode import odeint

config.update("jax_enable_x64", True)


class Box:
    """ Box class which represents each individual box in the carbon model."""

    def __init__(self, name, reservoir, production_coefficient=0.0, hemisphere='None'):
        """ init method to initialise object.
        Parameters
        ----------
        name : str
            name of the Box.
        reservoir : float
            reservoir content of the Box.
        production_coefficient : float, optional
            production coefficient of the Box. Defaults to 0.
        hemisphere : str, optional
            hemisphere that this box is in. Defaults to None (i.e. this box is not part of an inter-hemispheric system).
        """
        self._name = name
        self._reservoir = reservoir
        self._production = production_coefficient
        self._hemisphere = hemisphere
        assert reservoir >= 0
        assert production_coefficient >= 0
        assert hemisphere in ['None', 'north', 'south']

    def get_hemisphere(self):
        """ Getter method for the hemisphere of the Box Class.
        Returns
        -------
        str
            hemisphere the box is in.
        """
        return self._hemisphere

    def get_name(self):
        """ Getter method for the name of the Box Class.
        Returns
        -------
        str
            name of the Box.
        """
        return self._name

    def get_reservoir_content(self):
        """ Getter method for the reservoir content of the Box Class.
        Returns
        -------
        float
            reservoir content of the box.
        """
        return self._reservoir

    def get_production(self):
        """ Getter method for the production coefficient of the Box Class.
        Returns
        -------
         float
            production coefficient of the box.
        """
        return self._production

    def __str__(self):
        """ Overrides the default string behaviour to display a user-friendly output.
        Returns
        -------
        str
            string representation of the Box Object returned in the following form -
            name:reservoir size:production coefficient
        """
        return self._name + ":" + str(self._reservoir) + ":" + str(self._production)


class Flow:
    """ Flow class to imitate the fluxes between boxes in a carbon box model."""

    def __init__(self, source, destination, flux_rate):
        """ Init method to initialise object.
        Parameters
        ----------
        source : Box
            source of the flow.
        destination : Box
            destination the flow.
        flux_rate : float
            flux rate between the source and the destination. Flux must be a non-negative value.
        """
        self._source = source
        self._destination = destination
        assert flux_rate >= 0
        self._flux = flux_rate

    def get_source(self):
        """ Getter method for the source node of the Flow Class.
        Returns
        -------
        Box
            source of flow.
        """
        return self._source

    def get_destination(self):
        """ Getter method for the destination node of the Flow Class.
        Returns
        -------
        Box
            destination of the flow.
        """
        return self._destination

    def get_flux(self):
        """ Getter method for the flux rate of the Flow Class.
        Returns
        -------
        float
            flux of the flow object.
        """
        return self._flux

    def __str__(self):
        """ Overrides the default string behaviour to display a user-friendly output.
        Returns
        -------
        str
            string representation of the Flow Object returned in the following form -
            str(source)-->str(destination):flux_value
        """
        return str(self._source) + " --> " + str(self._destination) + " : " + str(self._flux)


class CarbonBoxModel:
    """Carbon Box Model class which represents the box model which is made up of Box Objects and Flux Objects."""

    def __init__(self, production_rate_units='kg/yr', flow_rate_units='Gt/yr'):
        """ Init method for creating an object of this class.
        Parameters
        ----------
        production_rate_units : str, optional
            units for the production rate. Only valid units are 'kg/yr' or 'atoms/cm^2/s'.
        flow_rate_units : str, optional
            units for the flow rate. Only valid values are 'Gt/yr' or '1/yr'.
        """
        self._non_hemisphere_model = False
        self._nodes = {}
        self._reverse_nodes = {}
        self._edges = []
        self._n_nodes = 0
        self._reservoir_content = None
        self._fluxes = None
        self._decay_matrix = None
        self._production_coefficients = None
        self._decay_constant = jnp.log(2) / 5700
        self._production_rate_units = production_rate_units
        self._flow_rate_units = flow_rate_units
        self._corrected_fluxes = None
        self._matrix = None

    def add_nodes(self, nodes):
        """ Adds the nodes to the Carbon Box Model. If the node already exists within the carbon box model node list,
        then it is not added. If the node is not a Box Class Instance, then it raises a ValueError.
        Parameters
        ----------
        nodes : list | ndarray
            A list of nodes of Type Box to add to the carbon box model.
        Raises
        ------
        ValueError
            If any of the objects in the list are not of type Box.
        """
        for node in nodes:
            if isinstance(node, Box):
                if node in self._nodes.values():
                    print(node.get_name() + " already exists in the Carbon Box. It was ignored and not "
                                            "added to existing nodes.")
                else:
                    self._nodes[self._n_nodes] = node
                    self._reverse_nodes[node] = self._n_nodes
                    self._n_nodes += 1
            else:
                raise ValueError("One/many of the input nodes are not of Box Class.")

    def add_edges(self, flow_objs):
        """ Adds the flow objects specified in the list to the carbon box model. If any of the objects in the list are
            not an instance of Flow Class then it throws a ValueError.
        Parameters
        ----------
        flow_objs : list | ndarray
            A list of Flow objects to add to the Carbon Box Model.
        Raises
        ------
        ValueError
            If any of the objects in the list are not of type Flow.
        """
        for flow_obj in flow_objs:
            if not isinstance(flow_obj, Flow):
                raise ValueError("One/many of the input edge are not of Flow Class.")
            self._edges.append(flow_obj)

    def get_edges(self):
        """ Getter method for the name of edges.
        Returns
        -------
        list
            A list of the edges (given in their string representations i.e. str(source) --> str(destination):flux_value).
        """
        return list(map(str, self._edges))

    def get_edges_objects(self):
        """ Getter method for the edge objects themselves.
        Returns
        -------
        list
            A list of Flow Objects that have been added so far to the Class Object.
        """
        return self._edges

    def get_nodes(self):
        """ Getter method for the name of the nodes in the order they were added to the Carbon Box Model Object.
        Returns
        -------
        list
            A list of node names in the order they were inserted.
        """
        return [self._nodes[j].get_name() for j in range(self._n_nodes)]

    def get_nodes_objects(self):
        """ Getter method for the node objects in the order they were inserted.
        Returns
        -------
        list
            A list of the node objects in the order they were inserted.
        """
        return [self._nodes[j] for j in range(self._n_nodes)]

    def get_fluxes(self):
        """ Getter method for the compiled fluxes in the units specified in the init_method. If the compile method has
            not been run, then it will return None.
        Returns
        -------
        DeviceArray
            2D jax numpy array which contains the fluxes where index (i,j) indicates the flux from node[i] to node[j].
        """
        return self._fluxes

    def get_converted_fluxes(self):
        """ Getter method for the fluxes when converted to 'Gt/yr' (this is the unit that the rest of the methods work
            in internally). This returns None if the compile method has not been run.
        Returns
        -------
        DeviceArray
            2D jax numpy array which contains the unit-corrected fluxes where index (i,j) indicates the 'Gt/yr' flux
            from node[i] to node[j].
        """
        return self._corrected_fluxes

    def get_reservoir_contents(self):
        """ Getter method for the 12C reservoir content of the nodes (Boxes).. Returns None if the compile method has
        not been called.
        Returns
        -------
        DeviceArray
            jax numpy array containing the reservoir content of the nodes (returned in the order the nodes were added).
        """
        return self._reservoir_content

    def get_production_coefficients(self):
        """ Getter method for the normalised production coefficients of the nodes (Boxes). Returns None if compile method
        has not been called.
        Returns
        -------
        DeviceArray
            jax array containing the normalised production coefficients of the nodes (returned in the order the
            nodes were added).
        """
        return self._production_coefficients


    def get_matrix(self):
        """ Getter method for the ODE coefficient matrix to solve.
        Returns
        -------
        DeviceArray
            jax array of the ODE coefficient matrix to solve.
        """
        return self._matrix

    @partial(jit, static_argnums=0)
    def _convert_production_rate(self, production_rate):
        if self._production_rate_units == 'atoms/cm^2/s':
            new_rate = production_rate * 14.003242 / 6.022 * 5.11 * 31536. / 1.e5
        elif self._production_rate_units == 'kg/yr':
            new_rate = production_rate
        else:
            raise ValueError('Production Rate units must be either atoms/cm^2/s or kg/yr!')
        return new_rate

    def _convert_flux_rate(self, fluxes):
        if self._flow_rate_units == 'Gt/yr':
            corrected_fluxes = self._fluxes
        elif self._flow_rate_units == '1/yr':
            corrected_fluxes = fluxes * jnp.transpose(self._reservoir_content) * 14.003242 / 12
            corrected_fluxes = corrected_fluxes + jnp.transpose(corrected_fluxes)
        else:
            raise ValueError('Flow rate units must be either Gt/yr or 1/yr!')
        return corrected_fluxes

    def compile(self):
        """ Method which compiles crucial parts of the model. If the model has not been compiled before then it
        compiles the following quantities:
        - 12C reservoir content of the nodes (in the order the nodes were added)
        - the fluxes (where fluxes at index (i,j) represents the flux from node[i] to node[j])
        - the decay matrix (matrix with decay constant along the diagonal)
        - production coefficients of the nodes.
        - the corrected fluxes (fluxes in unit 'Gt/yr')
        - the matrix of the coefficients for the ODEINT to solve.
        It also detects if the incoming and outgoing fluxes at every node is balanced and if not then throws ValueError
        along with which node is unbalanced.
        Raises
        ------
        ValueError
            If the incoming flux and outgoing flux at every Box is not balanced.
        """

        if self._fluxes is None:

            self._reservoir_content = jnp.array(
                [[self._nodes[j].get_reservoir_content() for j in range(self._n_nodes)]])
            self._fluxes = jnp.zeros((self._n_nodes, self._n_nodes))
            for flow in self._edges:
                self._fluxes = self._fluxes.at[self._reverse_nodes[flow.get_source()], 
                    self._reverse_nodes[flow.get_destination()]].set(flow.get_flux())

            self._decay_matrix = jnp.diag(jnp.array([self._decay_constant] * self._n_nodes))
            self._production_coefficients = jnp.array([self._nodes[j].get_production() for j in range(self._n_nodes)])
            self._production_coefficients /= jnp.sum(self._production_coefficients)
            self._corrected_fluxes = self._convert_flux_rate(self._fluxes)
            c_14_fluxes = self._corrected_fluxes / jnp.transpose(self._reservoir_content)
            new_c_14_fluxes = jnp.diag(jnp.sum(c_14_fluxes, axis=1))
            self._matrix = jnp.transpose(c_14_fluxes) - new_c_14_fluxes - self._decay_matrix
            self._non_hemisphere_model = self._nodes[0].get_hemisphere() == "None"

            for i in range(self._n_nodes):
                if jnp.abs(jnp.sum(self._corrected_fluxes[:, i]) - jnp.sum(self._corrected_fluxes[i, :])) > 0.001:
                    raise ValueError('the outgoing and incoming fluxes are not balanced for ' + str(self._nodes[i]))
                if self._non_hemisphere_model:
                    if not self._nodes[i].get_hemisphere() == "None":
                        raise ValueError(str(self._nodes[i]) + ' has been given a hemisphere value when there are '
                                                               'others which are None!')
                else:
                    if not self._nodes[i].get_hemisphere() in ['north', 'south']:
                        raise ValueError('There are some nodes where hemisphere is given and others where it is not. '
                                         'This is not allowed either all must be None of all of them must be either '
                                         '"north" or "south"')

    def _equilibrate_brehm(self, production_rate):
        production_rate = self._convert_production_rate(production_rate)
        solution = jnp.linalg.solve(self._matrix, -1 * self._production_coefficients * production_rate)
        return solution

    def _equilibrate_guttler(self, target_C_14):
        troposphere_index = None
        for index, node in self._nodes.items():
            if node.get_name() == 'Troposphere':
                troposphere_index = index
                break
        if troposphere_index is None:
            raise ValueError('there is currently no Troposphere node to equilibrate!')

        @jit
        def objective_function(production_rate):
            """ Function which calculates the difference between the current tropospheric C14 content and the target
            tropospheric C14 content.
            Parameters
            ----------
            production_rate : float
            Returns
            -------
            float
                The difference between the target and the equilibrium production.
            """
            equilibrium = self._equilibrate_brehm(production_rate)
            return (equilibrium[troposphere_index] - target_C_14) ** 2

        grad_obj = jax.jit(jax.grad(objective_function))
        final_production_rate = scipy.optimize.minimize(objective_function, jnp.array([6.]), method='BFGS',
                                                        jac=grad_obj)
        return final_production_rate.x[0]

    def equilibrate(self, target_C_14=None, production_rate=None):
        """ External equilibrate method which determines the appropriate result to return given a parameter. If
        neither parameter is given then it throws a ValueError. If both are specified, then it ignores production_rate.
        Parameters
        ----------
        target_C_14 : float
            target C14 with which to equilibrate with. Defaults to None.
            
        production_rate : float
            production rate with which to equilibrate to. Defaults to None.
        Returns
        -------
        float | ndarray
            if target C14 is specified, it returns the initial production rate. If the production rate is specified,
            then it returns a ndarray of the C14 reservoir content.
        Raises
        ------
        ValueError
            If the both arguments are specified as None.
        """

        if target_C_14 is not None:
            return self._equilibrate_guttler(target_C_14)

        elif production_rate is not None:
            return self._equilibrate_brehm(production_rate)

        else:
            raise ValueError("Must give either target C-14 or production rate.")

    @partial(jit, static_argnums=(0, 2, 5, 6))
    def run(self, time, production, y0=None, args=(), target_C_14=None, steady_state_production=None, solution=None):
        """ For the given production function, this calculates the C14 content of all the boxes within the carbon box
        model at the specified time values. It does this by solving a linear system of ODEs. This method will not work
        if the compile method has not been executed first.
        Parameters
        ----------
        time : list
            the time values at which to calculate the content of all the boxes.
        production : callable
            the production function which determines the contents of the boxes.
        y0 : list, optional
            the initial contents of all boxes. Defaults to None. Must be a length(n) list of the same size as the number
            of boxes.
        args : tuple, optional
            optional arguments to pass into the production function.
        steady_state_production : int, optional
            the steady state production rate with which to equilibrate with to find steady state solution. If y0 is
            specified, this parameter is ignored.
        target_C_14 : int, optional
            target C14 with which to equilibrate with to find steady state solution. If y0 or steady_state_production is
             specified, this parameter is ignored.
        solution : ndarray, optional
            the equilibrium solution to the ODE system. If this is not specified, then it will be calculated internally.
            Must be of the same length as the number of boxes.
        Returns
        -------
        Union[list, list]
            The value of each box in the carbon box at the specified time_values along with the steady state solution
            for the system.
        Raises
        ------
        ValueError
            If neither the target C-14 nor production rate is specified.
            If the production is not a callable function.
        """

        @jit
        def derivative(y, t):
            ans = jnp.matmul(self._matrix, y)
            production_rate_constant = production(t, *args) - steady_state_production
            production_rate_constant = self._convert_production_rate(production_rate_constant)
            production_term = self._production_coefficients * production_rate_constant
            return ans + production_term

        time_values = jnp.array(time)
        if solution is None:
            if steady_state_production is not None:
                solution = self.equilibrate(production_rate=steady_state_production)

            elif target_C_14 is not None:
                steady_state_production = self.equilibrate(target_C_14=target_C_14)
                solution = self.equilibrate(production_rate=steady_state_production)
            else:
                ValueError("Must give either target C-14 or production rate or steady state values of system.")

        if not callable(production):
            raise ValueError("incorrect object type for production")
            
        if y0 is not None:
            y_initial = jnp.array(y0)
        else:
            y_initial = jnp.array(solution)

        states = odeint(derivative, y_initial-solution, time_values,  atol=1e-15, rtol=1e-15) + solution
        return states, solution

    @partial(jit, static_argnums=(0, 2))
    def bin_data(self, data, time_oversample, time_out, growth):
        """ Bins the data given based on the oversample and the growth season according to Schulman's convention.
        Can handle any contiguous growth season, even over the year.
        Parameters
        ----------
        data : ndarray
            the data which to bin.
        time_oversample : int
            number of samples taken per year.
        time_out : ndarray
            the time values at which to bin the data at.
        growth : ndarray
            the growth season with which to bin the data with respect to.
        Returns
        -------
        DeviceArray
            The final binned data accounting for both growth season and Schulman's convention.
        Raises
        ------
        ValueError
            If the data is not one-dimensional or in a single row.
        """
        if data.ndim != 1:
            raise ValueError("Data is not one-dimensional! Data must be contained in one row. ")

        masked = jnp.linspace(0, 1, time_oversample)
        kernel = (masked < jnp.count_nonzero(growth)/12)
        
        @partial(jit)
        def _shifted_index_finder(seasons):
            first1 = jnp.where(seasons == 1, size=1)[0][0]
            first0 = jnp.where(seasons == 0, size=1)[0][0]
            all1s = jnp.where(seasons == 1, size=12)[0]
            after1 = jnp.where(all1s > first0, all1s, 0)
            after1 = after1.at[jnp.nonzero(after1, size=1)].get()[0]
            num = jax.lax.sub(first1, after1)
            val = cond(num == 0, lambda x: first1, lambda x : after1, num)
            act = cond(jnp.all(seasons == 1), lambda x: 0, lambda x: val, seasons)
            return act
    
        shifted_index = _shifted_index_finder(growth)
  
        binned_data = self._rebin1D(time_out, shifted_index, time_oversample, kernel, data)
        return binned_data

    @partial(jit, static_argnums=(0, 3))
    def _rebin1D(self, time_out, shifted_index, oversample, kernel, s):
        binned_data = jnp.zeros((len(time_out),))
        fun = lambda i, val: dynamic_update_slice(val, jnp.array([jnp.sum(jnp.multiply(dynamic_slice(
            s, (i * oversample + shifted_index * oversample // 12,), (oversample,)), kernel)) / (
                                                                      jnp.sum(kernel))]), (i,))
        binned_data = fori_loop(0, len(time_out), fun, binned_data)
        return binned_data


def save_model(carbon_box_model, filename):
    """ Saves the Carbon Box Model in a hd5 format with the specified filename. If the first parameter is not of Type
    CarbonBoxModel, then it throws a ValueError.
    Parameters
    ----------
    carbon_box_model : CarbonBoxModel
        the model to save.
    filename : str
        file name and location where Carbon Box Model needs to be saved. Must have a '.hd5' at end of filename.
    Raises
    ------
    ValueError
        If the first parameter is not of type CarbonBoxModel.
    """
    file = h5py.File(filename, 'a')
    file.clear()  # overwrites if it already exists
    if isinstance(carbon_box_model, CarbonBoxModel):
        metadata = {'fluxes': carbon_box_model.get_converted_fluxes(),
                    'reservoir content': carbon_box_model.get_reservoir_contents(),
                    'production coefficients': carbon_box_model.get_production_coefficients(),
                    'nodes': carbon_box_model.get_nodes(),
                    'hemispheres': [i.get_hemisphere() for i in carbon_box_model.get_nodes_objects()]}
    else:
        raise ValueError("parameter is not a Carbon Box Model!")

    hdfdict.dump(metadata, file)
    file.close()


def load_model(filename, production_rate_units='kg/yr', flow_rate_units='Gt/yr'):
    """ Loads the saved Carbon Box Model from the relevant filename. Units for both production rate and flow rate can be
    specified as parameters. filename must be specified with the .hd5 extension.
    Parameters
    ----------
    filename : str
        the name and location (in one string) of the file where the Carbon Box Model is saved.
    production_rate_units : str, optional
        the production rate of the model to be loaded. Defaults to 'kg/yr'.
    flow_rate_units : str, optional
        the production rate of the model to be loaded. Defaults to 'Gt/yr'.
    Returns
    -------
    CarbonBoxModel
        Carbon Box Model which is generated from the file.
    Raises
    ------
    ValueError
        If the flow rate units are not either 'Gt/yr' or '1/yr'.
    """

    file = h5py.File(filename, 'r')
    metadata = dict(hdfdict.load(file))
    nodes = metadata['nodes']
    hemispheres = metadata['hemispheres']
    carbon_box_model = CarbonBoxModel(production_rate_units=production_rate_units, flow_rate_units=flow_rate_units)
    box_object_list = []
    for j in range(len(nodes)):
        box_object_list.append(Box(nodes[j].decode("utf-8"), float(metadata['reservoir content'][0][j]),
                                   float(metadata['production coefficients'][j]), hemispheres[j].decode("utf-8")))

    carbon_box_model.add_nodes(box_object_list)

    for k in range(metadata['fluxes'].shape[0]):
        for j in range(metadata['fluxes'].shape[1]):
            if metadata['fluxes'][k, j] != 0:
                if flow_rate_units == 'Gt/yr':
                    carbon_box_model.add_edges([Flow(box_object_list[k], box_object_list[j],
                                                     float(metadata['fluxes'][k, j]))])

                elif flow_rate_units == '1/yr':
                    new_flow = float(metadata['fluxes'][k, j]) * 12 / 14.003242 / box_object_list[
                        k].get_reservoir_content()
                    carbon_box_model.add_edges([Flow(box_object_list[k], box_object_list[j], new_flow)])
                else:
                    raise ValueError('flow_rate_units are not valid.')

    file.close()
    return carbon_box_model


def load_presaved_model(model, production_rate_units='kg/yr', flow_rate_units='Gt/yr'):
    """ Loads a pre-saved, commonly used model based on the research papers linked below. The model must be one of the
    following: Miyake17, Brehm21, Guttler15, Buntgen18. Loads the model based on the units for production rate
    and flow rate specified.
    Parameters
    ----------
    model : str
        the name of the model to load. Must be one in [Miyake17, Brehm21, Guttler15, Buntgen18].
    production_rate_units : str, optional
        the production rate of the model to be loaded. Defaults to 'kg/yr'.
    flow_rate_units : str, optional
        the production rate of the model to be loaded. Defaults to 'Gt/yr'.
    Returns
    -------
    CarbonBoxModel
        Carbon Box Model which is generated from the pre-saved file.
    Raises
    ------
    ValueError
        If the specified model parameter is not in the required list.
    """
    if model in ['Guttler15', 'Brehm21', 'Miyake17', 'Buntgen18']:
        file = 'data/' + model + '.hd5'
        carbonmodel = load_model(pkg_resources.resource_stream(__name__, file),
                                 production_rate_units=production_rate_units, flow_rate_units=flow_rate_units)
        return carbonmodel
    else:
        raise ValueError('model parameter must be one of the following: Guttler15, Brehm21, Miyake17, Buntgen18')
