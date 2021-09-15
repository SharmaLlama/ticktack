import h5py
import hdfdict
import jax.numpy as jnp
import jax.ops
import scipy as scipy
import scipy.integrate
import scipy.optimize
from jax import jit, partial
from jax.config import config
import numpy as np
import pkg_resources

USE_JAX = True
if USE_JAX:
    from jax.experimental.ode import odeint
else:
    from scipy.integrate import odeint

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
        init method for when the Flow object is created.

        :param source: source of the flow.
        :param destination: destination the flow.
        :param flux_rate: flux rate between the source and the destination. Flux must be a non-negative value.
        """
        self._source = source
        self._destination = destination
        assert flux_rate >= 0
        self._flux = flux_rate

    def get_source(self):
        """
        getter method for the source node of the Flow Class.

        :return:
        returns the source of the flow.

        """
        return self._source

    def get_destination(self):
        """
        getter method for the destination node of the Flow Class.

        :return:
        returns the destination of the flow.
        """
        return self._destination

    def get_flux(self):
        """
        getter method for the flux rate of the Flow Class.

        :return:
        returns the flux of the flow.
        """
        return self._flux

    def __str__(self):
        """
        returns a string representation of the Flow.

        :return:
        returns the string in thw follow representation - str(source) --> str(destination):flux_value
        """
        return str(self._source) + " --> " + str(self._destination) + " : " + str(self._flux)


class CarbonBoxModel:
    def __init__(self, production_rate_units='kg/yr', flow_rate_units='Gt/yr'):
        """
        init method for when the Carbon Box Model object is created. Initialises the parameters that are needed for
        the rest of the class.

        :param production_rate_units: units for the production rate. Only valid values are 'kg/yr' or 'atoms/cm^2/s'.
        :param flow_rate_units: units for the flow rate. Only valid values are 'Gt/yr' or '1/yr'.
        """
        self._nodes = {}
        self._reverse_nodes = {}
        self._edges = []
        self._n_nodes = 0
        self._reservoir_content = None
        self._fluxes = None
        self._decay_matrix = None
        self._production_coefficients = None
        self._decay_constant = jnp.log(2) / 5730
        self._production_rate_units = production_rate_units
        self._flow_rate_units = flow_rate_units
        self._corrected_fluxes = None
        self._matrix = None
        self._growth_kernel = jnp.array([1] * 12)

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
        final_production_rate = scipy.optimize.minimize(objective_function, jnp.array([6.]), method='BFGS',
                                                        jac=grad_obj)
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
        def derivative(y, t):
            ans = jnp.matmul(self._matrix, y)
            # print("t shape: ", t.shape)
            # print("y: ", y)
            # print("t: ", t)
            # print("args: ", *args)
            production_rate_constant = production(t, *args)
            production_rate_constant = self._convert_production_rate(production_rate_constant)
            production_term = self._production_coefficients * production_rate_constant
            return ans + production_term

        time_values = jnp.array(time_values)
        # print("time_values shape: ", time_values.shape)
        solution = None
        if y0 is not None:
            y_initial = jnp.array(y0)
        else:
            if steady_state_production is not None:
                solution = self.equilibrate(production_rate=steady_state_production)

            elif target_C_14 is not None:
                solution = self.equilibrate(production_rate=self.equilibrate(target_C_14=target_C_14))
            else:
                ValueError("Must give either y0, or target C-14, or production rate.")
            y_initial = jnp.array(solution)

        if not callable(production):
            raise ValueError("incorrect object type for production")

        if USE_JAX:
            # print("time_values jax shape: ", time_values.shape)
            states = odeint(derivative, y_initial, time_values)
        else:
            # print("time_values no jax shape: ", time_values.shape)
            states = odeint(derivative, y_initial, time_values)
        return states, solution

    def run_bin(self, time_out, time_oversample, production, y0=None, args=(), target_C_14=None,
                steady_state_production=None):
        # time_out = jnp.array(time_out)
        # # time_step = time_out[1] - time_out[0]
        # t = jnp.linspace(jnp.min(time_out), jnp.max(time_out), (time_out.shape[0] - 1) * time_oversample)
        # states, solution = self.run(t, production, y0=y0, args=args, target_C_14=target_C_14,
        #                             steady_state_production=steady_state_production)
        # m = int(1 / time_step * time_oversample // 12)
        # tiled = jnp.resize(jnp.repeat(self._growth_kernel, m), (1,  int(time_oversample / time_step)))
        # num = int(time_oversample // (time_step * 12) * 12)
        # tiled = jax.ops.index_update(tiled, jnp.array([[False] * num + [True] * (tiled.shape[1] - num)]), 0)
        # tiled_full = jnp.resize(jnp.tile(tiled, (states.shape[1], int((time_out.shape[0] - 1) * time_step))),
        #                         (states.shape[1], states.shape[0]))
        # # print(tiled_full.shape)
        # # print(states)
        # states = jnp.transpose(tiled_full) * states
        # # print(jnp.reshape(states, (-1, states.shape[0] // time_oversample, time_oversample, states.shape[1])).sum(2))
        # binned_data = jnp.reshape(states, (-1, states.shape[0] // time_oversample, time_oversample, states.shape[1])) \
        #                   .sum(2).sum(0) / jnp.sum(tiled) / time_step

        time_out = jnp.array(time_out)
        t = jnp.linspace(jnp.min(time_out), jnp.max(time_out), (time_out.shape[0] - 1) * time_oversample)
        states, solution = self.run(t, production, y0=y0, args=args, target_C_14=target_C_14,
                                    steady_state_production=steady_state_production)
        m = time_oversample // 12
        tiled = jnp.resize(jnp.repeat(self._growth_kernel, m), (1,  time_oversample))
        num = int((time_oversample // 12) * 12)
        tiled = jax.ops.index_update(tiled, jnp.array([[False] * num + [True] * (tiled.shape[1] - num)]), 0)
        tiled_full = jnp.resize(jnp.tile(tiled, (states.shape[1], int(time_out.shape[0] - 1))),
                                (states.shape[1], states.shape[0]))
        states = jnp.transpose(tiled_full) * states
        binned_data = jnp.reshape(states, (-1, states.shape[0] // time_oversample, time_oversample, states.shape[1])) \
                          .sum(2).sum(0) / jnp.sum(tiled)

        return binned_data, solution

    def _to_d14c(self,data,solution):

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


    def run_D_14_C_values(self, time_out, time_oversample, production, y0=None, args=(), target_C_14=None,
                          steady_state_production=None, steady_state_solutions=None):
        time_out = jnp.array(time_out)
        data, soln = self.run_bin(time_out=time_out, time_oversample=time_oversample, production=production,
                                  y0=y0, args=args, target_C_14=target_C_14,
                                  steady_state_production=steady_state_production)

        if steady_state_solutions is None:
            solution = soln
        else:
            solution = steady_state_solutions

        return self._to_d14c(data,solution)

    def define_growth_season(self, months):
        month_list = np.array(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
                               'october', 'november', 'december'])
        months = np.array(months)
        self._growth_kernel = jax.ops.index_update(self._growth_kernel, np.in1d(month_list, months,
                                                                                invert=True), 0)

    def define_growth_season(self, months):
        month_list = np.array(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
                               'october', 'november', 'december'])
        months = np.array(months)
        self._growth_kernel = jax.ops.index_update(self._growth_kernel, np.in1d(month_list, months,
                                                                                invert=True), 0)


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
