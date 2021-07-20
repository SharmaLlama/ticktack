import h5py
import hdfdict
import jax.numpy as jnp
import jax.ops
import numpy as np
import scipy as scipy
import scipy.integrate
import scipy.optimize
from jax import jit
from jax.config import config
import pkg_resources

config.update("jax_enable_x64", True)


class Box:
    def __init__(self, name, reservoir, production_coefficient=0.0):
        """
        init method for when the object is created.

        :param name: name of the Box.
        :param reservoir: reservoir content of the Box.
        :param production_coefficient: production coefficient of the Box.
        """
        self._name = name
        self._reservoir = reservoir
        self._production = production_coefficient

    def get_name(self):
        """
        getter method for the name of the Box Class.

        :return:
            returns a string that represents the name of the Box.
        """
        return self._name

    def get_reservoir_content(self):
        """
        getter method for the reservoir content of the Box Class.

        :return:
            returns the reservoir content that was passed into __init__ method.
        """
        return self._reservoir

    def get_production(self):
        """
        getter method for the reservoir content of the Box Class.

        :return:
            returns the production coefficient that was passed into __init__ method.
        """
        return self._production

    def __str__(self):
        """
       returns a string representation of the Box.

        :return:
            returns the string in thw follow representation - name:reservoir_content:production_value
        """

        return self._name + ":" + str(self._reservoir) + ":" + str(self._production)


class Flow:
    def __init__(self, source, destination, flux_rate):
        """

        :param source:
        :param destination:
        :param flux_rate:
        """
        self._source = source
        self._destination = destination
        self._flux = flux_rate

    def get_source(self):
        """

        :return:
        """
        return self._source

    def get_destination(self):
        """

        :return:
        """
        return self._destination

    def get_flux(self):
        """

        :return:
        """
        return self._flux

    def __str__(self):
        """

        :return:
        """
        return str(self._source) + " --> " + str(self._destination) + " : " + str(self._flux)


class CarbonBoxModel:
    def __init__(self, production_rate_units='kg/yr', flow_rate_units='Gt/yr'):
        """

        :param production_rate_units:
        :param flow_rate_units:
        """
        self._nodes = {}
        self._reverse_nodes = {}
        self._edges = []
        self._n_nodes = 0
        self._reservoir_content = None
        self._fluxes = None
        self._decay_matrix = None
        self._production_coefficients = None
        self._decay_constant = jax.numpy.log(2) / 5730
        self._production_rate_units = production_rate_units
        self._flow_rate_units = flow_rate_units
        self._corrected_fluxes = None
        self._matrix = None

    def add_nodes(self, nodes):
        """
        
        :param nodes:
        :return:
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
        """

        :param flow_objs:
        :return:
        """
        for flow_obj in flow_objs:
            if not isinstance(flow_obj, Flow):
                raise ValueError("One/many of the input edge are not of Flow Class.")
            self._edges.append(flow_obj)

    def get_edges(self):
        """

        :return:
        """
        return list(map(str, self._edges))

    def get_edges_objects(self):
        """

        :return:
        """
        return self._edges

    def get_nodes(self):
        """

        :return:
        """
        return [self._nodes[j].get_name() for j in range(self._n_nodes)]

    def get_nodes_objects(self):
        """

        :return:
        """
        return [self._nodes[j] for j in range(self._n_nodes)]

    def get_fluxes(self):
        """

        :return:
        """
        return self._fluxes

    def get_converted_fluxes(self):
        """

        :return:
        """
        return self._corrected_fluxes

    def get_reservoir_contents(self):
        """

        :return:
        """
        return self._reservoir_content

    def get_production_coefficients(self):
        """

        :return:
        """
        return self._production_coefficients

    def _convert_production_rate(self, production_rate):
        if self._production_rate_units == 'atoms/cm^2/s':
            production_rate = production_rate * 14.003242 / 6.022 * 5.11 * 31536 / 10 ** 5
        elif self._production_rate_units == 'kg/yr':
            production_rate = production_rate
        else:
            raise ValueError('Production Rate units must be either atoms/cm^2/s or kg/yr!')
        return production_rate

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
        if self._fluxes is None:
            self._reservoir_content = jnp.array(
                [[self._nodes[j].get_reservoir_content() for j in range(self._n_nodes)]])
            self._fluxes = jnp.zeros((self._n_nodes, self._n_nodes))
            for flow in self._edges:
                self._fluxes = jax.ops.index_update(self._fluxes,
                                                    jax.ops.index[self._reverse_nodes[flow.get_source()],
                                                                  self._reverse_nodes[flow.get_destination()]],
                                                    flow.get_flux())

            self._decay_matrix = jnp.diag(jnp.array([self._decay_constant] * self._n_nodes))
            self._production_coefficients = jnp.array([self._nodes[j].get_production() for j in range(self._n_nodes)])
            self._production_coefficients /= jnp.sum(self._production_coefficients)
            self._corrected_fluxes = self._convert_flux_rate(self._fluxes)
            c_14_fluxes = self._corrected_fluxes / jnp.transpose(self._reservoir_content)
            new_c_14_fluxes = jnp.diag(jnp.sum(c_14_fluxes, axis=1))
            self._matrix = jnp.transpose(c_14_fluxes) - new_c_14_fluxes - self._decay_matrix

            for i in range(self._n_nodes):
                if jnp.abs(jnp.sum(self._corrected_fluxes[:, i]) - jnp.sum(self._corrected_fluxes[i, :])) > 0.001:
                    raise ValueError('the outgoing and incoming fluxes are not balanced for ' + str(self._nodes[i]))

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
            equilibrium = self._equilibrate_brehm(production_rate)
            return (equilibrium[troposphere_index] - target_C_14) ** 2

        grad_obj = jax.jit(jax.grad(objective_function))
        final_production_rate = scipy.optimize.minimize(objective_function, np.array([6.]), method='BFGS', jac=grad_obj)
        return final_production_rate.x[0]

    def equilibrate(self, target_C_14=None, production_rate=None):
        if target_C_14 is not None:
            return self._equilibrate_guttler(target_C_14)

        elif production_rate is not None:
            return self._equilibrate_brehm(production_rate)

        else:
            raise ValueError("Must give either target C-14 or production rate.")

    def run(self, time_values, production, y0=None, args=(), target_C_14=None, steady_state_production=None):

        @jit
        def derivative(y, t, prod_val):
            ans = jnp.matmul(self._matrix, y)
            production_rate_constant = (lambda x: jnp.interp(x, time_values, prod_val))(t)
            production_term = self._production_coefficients * production_rate_constant
            return ans + production_term

        time_values = np.array(time_values)
        solution = None
        if y0 is not None:
            y_initial = y0
        else:
            if steady_state_production is not None:
                solution = self.equilibrate(production_rate=steady_state_production)

            elif target_C_14 is not None:
                solution = self.equilibrate(production_rate=self.equilibrate(target_C_14=target_C_14))
            else:
                ValueError("Must give either target C-14 or production rate.")
            y_initial = solution

        if callable(production):
            production_array = jnp.array([production(time_values[j], *args) for j in range(time_values.shape[0])])

        else:
            if isinstance(production, (float, int)):
                production_array = production * jnp.ones_like(time_values)

            elif isinstance(production, (np.ndarray, list, jnp.ndarray)):
                production_array = jnp.array(production)

            elif not time_values.shape == production.shape:
                raise ValueError("time array and production array have different sizes")

            else:
                raise ValueError("incorrect object type for production")

        production_array = self._convert_production_rate(production_array)
        states = scipy.integrate.odeint(derivative, y_initial, time_values, args=(production_array,))
        return states, solution

    def run_bin(self, time_out, time_oversample, production, y0=None, args=(), target_C_14=None,
                steady_state_production=None):
        time_out = np.array(time_out)
        t = np.linspace(np.min(time_out), np.max(time_out), (time_out.shape[0] - 1) * time_oversample)
        states, solution = self.run(t, production, y0=y0, args=args, target_C_14=target_C_14,
                                    steady_state_production=steady_state_production)
        binned_data = np.reshape(states, (-1, states.shape[0] // time_oversample, time_oversample, states.shape[1])) \
                          .sum(2).sum(0) / time_oversample

        return binned_data, solution

    def run_D_14_C_values(self, time_out, time_oversample, production, y0=None, args=(), target_C_14=None,
                          steady_state_production=None, steady_state_solutions=None):

        time_out = np.array(time_out)
        data, soln = self.run_bin(time_out=time_out, time_oversample=time_oversample, production=production,
                                  y0=y0, args=args, target_C_14=target_C_14,
                                  steady_state_production=steady_state_production)

        if steady_state_solutions is None:
            solution = soln

        else:
            solution = steady_state_solutions

        troposphere_steady_state = None
        d_14_c = None
        for index, node in self._nodes.items():
            if node.get_name() == 'Troposphere':
                troposphere_steady_state = solution[index]
                d_14_c = (data[:, index] - troposphere_steady_state) / troposphere_steady_state * 1000
                break
        if troposphere_steady_state is None:
            raise ValueError('there is currently no Troposphere node to equilibrate!')
        return d_14_c


def save_model(carbon_box_model, filename):
    file = h5py.File(filename, 'a')
    file.clear()  # overwrites if it already exists
    metadata = None
    if isinstance(carbon_box_model, CarbonBoxModel):
        metadata = {'fluxes': carbon_box_model.get_converted_fluxes(),
                    'reservoir content': carbon_box_model.get_reservoir_contents(),
                    'production coefficients': carbon_box_model.get_production_coefficients(),
                    'nodes': carbon_box_model.get_nodes()}
    else:
        raise ValueError("parameter is not a Carbon Box Model!")

    hdfdict.dump(metadata, file)
    file.close()


def load_model(filename, production_rate_units='kg/yr', flow_rate_units='Gt/yr'):
    file = h5py.File(filename, 'r')
    metadata = dict(hdfdict.load(file))
    nodes = metadata['nodes']
    carbon_box_model = CarbonBoxModel(production_rate_units=production_rate_units, flow_rate_units=flow_rate_units)
    box_object_list = []
    for j in range(len(nodes)):
        box_object_list.append(Box(nodes[j].decode("utf-8"), float(metadata['reservoir content'][0][j]),
                                   float(metadata['production coefficients'][j])))

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
    if model in ['Guttler14', 'Brehm21', 'Miyake17', 'Buntgen18']:
        file = 'data/' + model + '.hd5'
        carbonmodel = load_model(pkg_resources.resource_stream(__name__, file),
                                 production_rate_units=production_rate_units, flow_rate_units=flow_rate_units)
        return carbonmodel
    else:
        raise ValueError('model parameter must be one of the following: Guttler14, Brehm21, Miyake17, Buntgen18')
