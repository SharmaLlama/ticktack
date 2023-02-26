import numpy as np
import jax.numpy as jnp
import pytest

import ticktack

import pandas


@pytest.fixture
def box1_creation():
    return ticktack.Box('troposphere', reservoir=100, production_coefficient=0.3)


@pytest.fixture()
def box2_creation():
    return ticktack.Box('marine surface', reservoir=150.14)


def test_get_name(box1_creation, box2_creation):
    assert box1_creation.get_name() == 'troposphere'
    assert box2_creation.get_name() == 'marine surface'


def test_get_reservoir_content(box1_creation, box2_creation):
    assert box1_creation.get_reservoir_content() == 100
    assert box2_creation.get_reservoir_content() == 150.14


def test_get_production_non_default(box1_creation):
    assert box1_creation.get_production() == 0.3


def test_get_production_default(box2_creation):
    assert box2_creation.get_production() == 0


def test_str(box1_creation, box2_creation):
    assert str(box1_creation) == 'troposphere:100:0.3'
    assert str(box2_creation) == 'marine surface:150.14:0.0'


@pytest.fixture
def flow_object_1_creation():
    box1 = ticktack.Box('troposphere', reservoir=100, production_coefficient=0.3)
    box2 = ticktack.Box('marine surface', reservoir=150.14)
    return box1, box2, ticktack.Flow(box1, box2, 66.2)


@pytest.fixture
def flow_object_2_creation():
    box1 = ticktack.Box('troposphere', reservoir=100, production_coefficient=0.3)
    box2 = ticktack.Box('marine surface', reservoir=150.14)
    return box1, box2, ticktack.Flow(box2, box1, 110.5)


def test_get_source(flow_object_1_creation, flow_object_2_creation):
    assert flow_object_1_creation[2].get_source() == flow_object_1_creation[0]
    assert flow_object_2_creation[2].get_source() == flow_object_2_creation[1]


def test_get_destination(flow_object_1_creation, flow_object_2_creation):
    assert flow_object_1_creation[2].get_destination() == flow_object_1_creation[1]
    assert flow_object_2_creation[2].get_destination() == flow_object_2_creation[0]


def test_get_flux(flow_object_1_creation, flow_object_2_creation):
    assert flow_object_1_creation[2].get_flux() == 66.2
    assert flow_object_2_creation[2].get_flux() == 110.5


def test_str_flow(flow_object_1_creation, flow_object_2_creation):
    actual = 'troposphere:100:0.3 --> marine surface:150.14:0.0 : 66.2'
    assert str(flow_object_1_creation[2]) == actual
    actual2 = 'marine surface:150.14:0.0 --> troposphere:100:0.3 : 110.5'
    assert str(flow_object_2_creation[2]) == actual2


@pytest.fixture
def cbm_object_creation():
    stra = ticktack.Box('Stratosphere', 60, 0.7)
    trop = ticktack.Box('Troposphere', 50, 0.3)
    ms = ticktack.Box("Marine surface", 900)
    bio = ticktack.Box("Biosphere", 1600)
    f1 = ticktack.Flow(stra, trop, 0.5)
    f2 = ticktack.Flow(trop, ms, 0.2)
    f3 = ticktack.Flow(trop, bio, 1)
    nodes = [stra, trop, ms, bio]
    edges = [f1, f2, f3]
    cbm = ticktack.CarbonBoxModel(flow_rate_units='1/yr')
    cbm.add_nodes(nodes)
    cbm.add_edges(edges)
    cbm.compile()
    return nodes, edges, cbm


def test_get_production_coefficients(cbm_object_creation):
    assert jnp.all(cbm_object_creation[2].get_production_coefficients() == np.array([0.7, 0.3, 0, 0]))


def test_get_edges(cbm_object_creation):
    actual = ['Stratosphere:60:0.7 --> Troposphere:50:0.3 : 0.5',
              'Troposphere:50:0.3 --> Marine surface:900:0.0 : 0.2',
              'Troposphere:50:0.3 --> Biosphere:1600:0.0 : 1']
    assert cbm_object_creation[2].get_edges() == actual


def test_get_edge_objects(cbm_object_creation):
    assert cbm_object_creation[2].get_edges_objects() == cbm_object_creation[1]


def test_get_nodes(cbm_object_creation):
    actual = ['Stratosphere', 'Troposphere', 'Marine surface', 'Biosphere']
    assert cbm_object_creation[2].get_nodes() == actual


def test_get_nodes_objects(cbm_object_creation):
    assert cbm_object_creation[2].get_nodes_objects() == cbm_object_creation[0]


def test_converted_fluxes_not_changed():
    box1 = ticktack.Box('no1', 30)
    box2 = ticktack.Box('no2', 40)
    box3 = ticktack.Box('no3', 50)
    flow1 = ticktack.Flow(box1, box2, 10)
    flow2 = ticktack.Flow(box2, box1, 10)
    flow3 = ticktack.Flow(box1, box3, 15)
    flow4 = ticktack.Flow(box3, box1, 15)
    cbm = ticktack.CarbonBoxModel()
    cbm.add_nodes([box1, box2, box3])
    cbm.add_edges([flow1, flow2, flow3, flow4])
    cbm.compile()
    actual = np.array([[0, 10, 15], [10, 0, 0], [15, 0, 0]])
    assert jnp.all(cbm.get_converted_fluxes() == actual)


def test_get_reservoir_contents(cbm_object_creation):
    actual = np.array([60, 50, 900, 1600])
    assert jnp.all(cbm_object_creation[2].get_reservoir_contents() == actual)


def test_get_fluxes(cbm_object_creation):
    actual = np.array([[0, 0.5, 0, 0], [0, 0, 0.2, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert jnp.all(cbm_object_creation[2].get_fluxes() == actual)


def test_converted_fluxes_changed(cbm_object_creation):
    actual = np.array([[0, 35.008105, 0, 0], [35.008105, 0, 11.6693683, 58.34684167], [0, 11.6693683, 0, 0],
                       [0, 58.34684167, 0, 0]])
    assert jnp.all(jnp.allclose(cbm_object_creation[2].get_converted_fluxes(), actual))


def test_add_node_not_box_class():
    cbm = ticktack.CarbonBoxModel()
    with pytest.raises(ValueError):
        cbm.add_nodes(['a'])

    assert True


def test_add_node_duplicate():
    cbm = ticktack.CarbonBoxModel()
    box1 = ticktack.Box('t1', 101)
    cbm.add_nodes([box1, box1, box1])
    assert cbm.get_nodes() == [box1.get_name()]


def test_add_edge_not_flow_class():
    cbm = ticktack.CarbonBoxModel()
    with pytest.raises(ValueError):
        cbm.add_edges(['a'])

    assert True


def test_add_edge_not_flow_class2():
    cbm = ticktack.CarbonBoxModel()
    b1 = ticktack.Box('t1', 101)
    with pytest.raises(ValueError):
        cbm.add_edges([b1])

    assert True


def test_run_bin_october_march():
    cbm = ticktack.load_presaved_model('Guttler15', production_rate_units='atoms/cm^2/s')
    cbm.compile()

    dates = np.linspace(774, 776, 100)
    t = np.linspace(773, 777, 4 * 300)
    binned = []
    actual = []

    def rebin(s):
        start_month_bin_index = 3
        time_out = [774, 775, 776]
        binned_data = np.array([0.0] * 3)
        oversample = 300

        for i in range(len(time_out)):
            chunk = s[(i + 1) * oversample - start_month_bin_index * oversample // 12:
                      (i + 2) * oversample - start_month_bin_index * oversample // 12]

            masked = np.linspace(0, 1, chunk.shape[0])
            kernel = 1.0 * (masked < 0.5)
            binned_data[i] = np.sum(chunk * kernel) / (np.sum(kernel))

        return binned_data[1]

    for date in dates:
        step = 1.0 * (t > date)
        a = rebin(step)
        b = cbm.bin_data(step, 300, jnp.array([774, 775, 776]), jnp.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]))[1]
        binned.append(a)
        actual.append(b)

    assert actual == binned


def test_run_bin_april_september():
    cbm = ticktack.load_presaved_model('Guttler15', production_rate_units='atoms/cm^2/s')
    cbm.compile()

    dates = np.linspace(774, 776, 100)
    t = np.linspace(773, 777, 4 * 300)
    binned = []
    actual = []

    def rebin(s):
        start_month_bin_index = 9
        time_out = [774, 775, 776]
        binned_data = np.array([0.0] * 3)
        oversample = 300

        for i in range(len(time_out)):
            chunk = s[(i + 1) * oversample - start_month_bin_index * oversample // 12:
                      (i + 2) * oversample - start_month_bin_index * oversample // 12]

            masked = np.linspace(0, 1, chunk.shape[0])
            kernel = 1.0 * (masked < 0.5)
            binned_data[i] = np.sum(chunk * kernel) / (np.sum(kernel))

        return binned_data[1]

    for date in dates:
        step = 1.0 * (t > date)
        a = rebin(step)
        b = cbm.bin_data(step, 300, jnp.array([774, 775, 776]), jnp.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]))[1]
        binned.append(a)
        actual.append(b)

    assert actual == binned
