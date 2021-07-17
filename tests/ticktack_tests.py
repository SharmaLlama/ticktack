import numpy as np
import jax.numpy as jnp

import pytest

from ticktack import Box, CarbonBoxModel, Flow


@pytest.fixture
def box1_creation():
    return Box('troposphere', reservoir=100, production_coefficient=0.3)


@pytest.fixture()
def box2_creation():
    return Box('marine surface', reservoir=150.14)


def test_get_name():
    assert box1_creation.get_name() == 'troposphere'
    assert box2_creation.get_name() == 'marine surface'


def test_get_reservoir_content():
    assert box1_creation.get_reservoir_content() == 100
    assert box2_creation.get_reservoir_content() == 150.14


def test_get_production_non_default():
    assert box1_creation.get_production() == 0.3


def test_get_production_default():
    assert box2_creation.get_production() == 44


def test_str():
    assert str(box1_creation) == 'troposphere:100:0.3'
    assert str(box2_creation) == 'marine surface:150.14:0.0'


@pytest.fixture
def flow_object_1_creation():
    box1 = Box('troposphere', reservoir=100, production_coefficient=0.3)
    box2 = Box('marine surface', reservoir=150.14)
    return box1, box2, Flow(box1, box2, 66.2)


def flow_object_2_creation():
    box1 = Box('troposphere', reservoir=100, production_coefficient=0.3)
    box2 = Box('marine surface', reservoir=150.14)
    return box1, box2, Flow(box2, box1, 110.5)


def test_get_source():
    assert flow_object_1_creation()[2].get_source() == flow_object_1_creation()[0]
    assert flow_object_2_creation()[2].get_source() == flow_object_2_creation()[0]


def test_get_destination():
    assert flow_object_1_creation()[2].get_destination() == flow_object_1_creation()[1]
    assert flow_object_2_creation()[2].get_destination() == flow_object_2_creation()[1]


def test_get_flux():
    assert flow_object_1_creation()[2].get_flux() == 66.2
    assert flow_object_2_creation()[2].get_flux() == 110.5


def test_str_flow():
    actual = 'troposphere:100:0.3 --> marine surface:150.14:0.0 : 66.2'
    assert str(flow_object_1_creation()[2]) == actual
    actual2 = 'marine surface:150.14:0.0 --> troposphere:100:0.3 : 110.5'
    assert str(flow_object_2_creation()[2]) == actual2


@pytest.fixture
def cbm_object_creation():
    stra = Box('Stratosphere', 60, 0.7)
    trop = Box('Troposphere', 50, 0.3)
    ms = Box("Marine surface", 900)
    bio = Box("Biosphere", 1600)
    f1 = Flow(stra, trop, 0.5)
    f2 = Flow(trop, ms, 0.2)
    f3 = Flow(trop, bio, 1)
    nodes = [stra, trop, ms, bio]
    edges = [f1, f2, f3]
    cbm = CarbonBoxModel(flow_rate_units='1/yr')
    cbm.add_nodes(nodes)
    cbm.add_edges(edges)
    cbm.compile()
    return nodes, edges, cbm


def test_get_production_coefficients():
    assert (jnp.all(cbm_object_creation[2].get_production_coefficients() == np.array([0.7, 0.3, 0, 0])), True)


def test_get_edges():
    actual = ['Stratosphere:60:0.7 --> Troposphere:50:0.3 : 0.5',
              'Troposphere:50:0.3 --> Marine surface:900:0.0 : 0.2',
              'Troposphere:50:0.3 --> Biosphere:1600:0.0 : 1']
    assert cbm_object_creation[2].get_edges() == actual


def test_get_edge_objects():
    assert cbm_object_creation[2].get_edges_objects() == cbm_object_creation[1]


def test_get_nodes():
    actual = ['Stratosphere', 'Troposphere', 'Marine surface', 'Biosphere']
    assert cbm_object_creation[2].get_nodes() == actual


def test_get_nodes_objects():
    assert cbm_object_creation[2].get_nodes_objects() == cbm_object_creation[0]


def test_converted_fluxes_not_changed():
    box1 = Box('no1', 30)
    box2 = Box('no2', 40)
    box3 = Box('no3', 50)
    flow1 = Flow(box1, box2, 10)
    flow2 = Flow(box2, box1, 10)
    flow3 = Flow(box1, box3, 15)
    flow4 = Flow(box3, box1, 15)
    cbm = CarbonBoxModel()
    cbm.add_nodes([box1, box2, box3])
    cbm.add_edges([flow1, flow2, flow3, flow4])
    cbm.compile()
    actual = np.array([[0, 10, 15], [10, 0, 0], [15, 0, 0]])
    assert (np.all(cbm.get_converted_fluxes() == actual), True)


def test_get_reservoir_contents():
    actual = np.array([60, 50, 900, 1600])
    assert (jnp.all(cbm_object_creation[2].get_reservoir_contents() == actual), True)


def test_get_fluxes():
    actual = np.array([[0, 0.5, 0, 0], [0, 0, 0.2, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert (jnp.all(cbm_object_creation[2].get_fluxes() == actual), True)


def test_converted_fluxes_changed():
    actual = np.array([[0, 35.008105, 0, 0], [35.008105, 0, 11.6693683, 58.34684167], [0, 11.6693683, 0, 0],
                       [0, 58.34684167, 0, 0]])
    assert (jnp.all(jnp.allclose(cbm_object_creation[2].get_converted_fluxes(), actual)), True)


def test_add_node_not_box_class():
    cbm = CarbonBoxModel()
    with pytest.raises(ValueError):
        cbm.add_nodes(['a'])

    assert False


def test_add_node_duplicate():
    cbm = CarbonBoxModel()
    box1 = Box('t1', 101)
    cbm.add_nodes([box1, box1, box1])
    assert cbm.get_nodes() == [box1.get_name()]


def test_add_edge_not_flow_class():
    cbm = CarbonBoxModel()
    with pytest.raises(ValueError):
        cbm.add_edges(['a'])

    assert False


def test_add_edge_not_flow_class2():
    cbm = CarbonBoxModel()
    b1 = Box('t1', 101)
    with pytest.raises(ValueError):
        cbm.add_edges([b1])

    assert False
