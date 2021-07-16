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

config.update("jax_enable_x64", True)


class Box:
    def __init__(self, name, reservoir, production_coefficient=0.0):
        self._name = name
        self._reservoir = reservoir
        self._production = production_coefficient

    def get_name(self):
        return self._name

    def get_reservoir_content(self):
        return self._reservoir

    def get_production(self):
        return self._production

    def __str__(self):
        return self._name


class Flow:
    def __init__(self, source, destination, flux_rate):
        self._source = source
        self._destination = destination
        self._flux = flux_rate

    def get_source(self):
        return self._source

    def get_destination(self):
        return self._destination

    def get_flux(self):
        return self._flux

    def __str__(self):
        return str(self._source) + " --> " + str(self._destination) + " : " + str(self._flux)


class CarbonBoxModel:
    def __init__(self):
        self._nodes = {}
        self._reverse_nodes = {}
        self._edges = []
        self._n_nodes = 0
        self._reservoir_content = None
        self._fluxes = None
        self._decay_matrix = None
        self._production_coefficients = None
        self._decay_constant = jax.numpy.log(2) / 5730

    def add_nodes(self, nodes):
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
        for flow_obj in flow_objs:
            if not isinstance(flow_obj, Flow):
                raise ValueError("One/many of the input edge are not of Flow Class.")
            self._edges.append(flow_obj)

    def get_edges(self):
        return list(map(str, self._edges))

    def get_edges_objects(self):
        return self._edges

    def get_nodes(self):
        return [self._nodes[j].get_name() for j in range(self._n_nodes)]

    def get_nodes_objects(self):
        return [self._nodes[j] for j in range(self._n_nodes)]

    def get_fluxes(self):
        return self._fluxes

    def get_reservoir_contents(self):
        return self._reservoir_content

    def get_production_coefficients(self):
        return self._production_coefficients

    def compile(self):
        if self._fluxes is None:
            self._reservoir_content = np.array([[self._nodes[j].get_reservoir_content() for j in range(self._n_nodes)]])
            self._fluxes = np.zeros((self._n_nodes, self._n_nodes))
            for flow in self._edges:
                self._fluxes = jax.ops.index_update(self._fluxes,
                                                    jax.ops.index[self._reverse_nodes[flow.get_source()],
                                                                  self._reverse_nodes[flow.get_destination()]],
                                                    flow.get_flux())
            for i in range(self._n_nodes):
                if jnp.abs(jnp.sum(self._fluxes[:, i]) - jnp.sum(self._fluxes[i, :])) > 0.001:
                    raise ValueError('the outgoing and incoming fluxes are not balanced for ' + str(self._nodes[i]))

            self._decay_matrix = jnp.diag(jnp.array([self._decay_constant] * self._n_nodes))
            self._production_coefficients = jnp.array([self._nodes[j].get_production() for j in range(self._n_nodes)])
            self._production_coefficients /= jnp.sum(self._production_coefficients)

    def _equilibrate_brehm(self, production_rate):
        c_14_fluxes = self._fluxes / jnp.transpose(self._reservoir_content)
        new_c_14_fluxes = jnp.diag(jnp.sum(c_14_fluxes, axis=1))
        matrix_to_solve = jnp.transpose(c_14_fluxes) - new_c_14_fluxes - self._decay_matrix
        solution = jnp.linalg.solve(matrix_to_solve, -1 * self._production_coefficients * production_rate)
        return matrix_to_solve, solution

    def _equilibrate_guttler(self, target_C_14):
        try:
            troposphere_index = self._reverse_nodes['troposphere']
        except KeyError:
            raise ValueError('there is currently no troposphere node to equilibrate!')

        @jit
        def objective_function(production_rate):
            _, equilibrium = self._equilibrate_brehm(production_rate)
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
            ans = jnp.matmul(matrix, y)
            production_rate_constant = (lambda x: jnp.interp(x, time_values, prod_val))(t)
            production_term = self._production_coefficients * production_rate_constant
            return ans + production_term

        if steady_state_production is not None:
            matrix, solution = self.equilibrate(production_rate=steady_state_production)

        elif target_C_14 is not None:
            matrix, solution = self.equilibrate(production_rate=self.equilibrate(target_C_14=target_C_14))

        else:
            raise ValueError("Must give either target C-14 or production rate.")

        y0 = y0 if y0 is not None else np.array(solution)
        time_values = np.array(time_values)

        if callable(production):
            production_array = np.array([production(time_values[j], *args) for j in range(time_values.shape[0])])

        else:
            if isinstance(production, (float, int)):
                production_array = production * np.ones_like(time_values)

            elif isinstance(production, (np.ndarray, list, jnp.ndarray)):
                production_array = np.array(production)

            elif not time_values.shape == production.shape:
                raise ValueError("time array and production array have different sizes")

            else:
                raise ValueError("incorrect object type for production")

        states = scipy.integrate.odeint(derivative, y0, time_values, args=(production_array,))
        return states

    def run_bin(self, time_out, time_oversample, production, y0=None, args=(), target_C_14=None,
                steady_state_production=None):

        time_out = np.array(time_out)
        t = np.linspace(np.min(time_out), np.max(time_out), (time_out.shape[0] - 1) * time_oversample)
        states = self.run(t, production, y0=y0, args=args, target_C_14=target_C_14,
                          steady_state_production=steady_state_production)
        binned_data = np.reshape(states, (-1, states.shape[0] // time_oversample, time_oversample, states.shape[1])) \
                          .sum(2).sum(0) / time_oversample

        return binned_data


def save_model(carbon_box_model, filename):
    file = h5py.File(filename, 'a')
    file.clear()  # overwrites if it already exists
    metadata = None
    if isinstance(carbon_box_model, CarbonBoxModel):
        metadata = {'fluxes': carbon_box_model.get_fluxes(),
                    'reservoir content': carbon_box_model.get_reservoir_contents(),
                    'production coefficients': carbon_box_model.get_production_coefficients(),
                    'nodes': carbon_box_model.get_nodes()}
    else:
        assert "parameter is not a Carbon Box Model!"

    hdfdict.dump(metadata, file)
    file.close()


def load_model(filename):
    file = h5py.File(filename, 'r')
    metadata = dict(hdfdict.load(file))
    nodes = metadata['nodes']
    carbon_box_model = CarbonBoxModel()
    box_object_list = []
    for j in range(len(nodes)):
        box_object_list.append(Box(nodes[j].decode("utf-8"), float(metadata['reservoir content'][0][j]),
                                   float(metadata['production coefficients'][j])))

    carbon_box_model.add_nodes(box_object_list)

    for k in range(metadata['fluxes'].shape[0]):
        for j in range(metadata['fluxes'].shape[1]):
            if metadata['fluxes'][k, j] != 0:
                carbon_box_model.add_edges([Flow(box_object_list[k], box_object_list[j],
                                                 float(metadata['fluxes'][k, j]))])

    file.close()
    return carbon_box_model
