import unittest

import numpy as np
import jax.numpy as jnp

from ticktack import Box, CarbonBoxModel, Flow, save_model, load_model


class TestBox(unittest.TestCase):
    def setUp(self) -> None:
        self.box1 = Box('troposphere', reservoir=100, production_coefficient=0.3)
        self.box2 = Box('marine surface', reservoir=150.14)

    def test_get_name(self):
        self.assertEqual(self.box1.get_name(), 'troposphere')
        self.assertEqual(self.box2.get_name(), 'marine surface')

    def test_get_reservoir_content(self):
        self.assertEqual(self.box1.get_reservoir_content(), 100)
        self.assertEqual(self.box2.get_reservoir_content(), 150.14)

    def test_get_production_non_default(self):
        self.assertEqual(self.box1.get_production(), 0.3)

    def test_get_production_default(self):
        self.assertEqual(self.box2.get_production(), 0)

    def test_str(self):
        self.assertEqual(str(self.box1), 'troposphere:100:0.3')
        self.assertEqual(str(self.box2), 'marine surface:150.14:0.0')


class TestFlow(unittest.TestCase):
    def setUp(self) -> None:
        self.box1 = Box('troposphere', reservoir=100, production_coefficient=0.3)
        self.box2 = Box('marine surface', reservoir=150.14)
        self.flow1 = Flow(self.box1, self.box2, 66.2)
        self.flow2 = Flow(self.box2, self.box1, 110.5)

    def test_get_source(self):
        self.assertEqual(self.flow1.get_source(), self.box1)
        self.assertEqual(self.flow2.get_source(), self.box2)

    def test_get_destination(self):
        self.assertEqual(self.flow1.get_destination(), self.box2)
        self.assertEqual(self.flow2.get_destination(), self.box1)

    def test_get_flux(self):
        self.assertEqual(self.flow1.get_flux(), 66.2)
        self.assertEqual(self.flow2.get_flux(), 110.5)

    def test_str(self):
        actual = 'troposphere:100:0.3 --> marine surface:150.14:0.0 : 66.2'
        self.assertEqual(str(self.flow1), actual)
        actual2 = 'marine surface:150.14:0.0 --> troposphere:100:0.3 : 110.5'
        self.assertEqual(str(self.flow2), actual2)


class TestTickTack(unittest.TestCase):

    def setUp(self) -> None:
        stra = Box('Stratosphere', 60, 0.7)
        trop = Box('Troposphere', 50, 0.3)
        ms = Box("Marine surface", 900)
        bio = Box("Biosphere", 1600)
        f1 = Flow(stra, trop, 0.5)
        f2 = Flow(trop, ms, 0.2)
        f3 = Flow(trop, bio, 1)
        self.nodes = [stra, trop, ms, bio]
        self.edges = [f1, f2, f3]
        self.cbm = CarbonBoxModel(flow_rate_units='1/yr')
        self.cbm.add_nodes([stra, trop, ms, bio])
        self.cbm.add_edges([f1, f2, f3])
        self.cbm.compile()

    def test_get_production_coefficients(self):
        self.assertTrue(jnp.all(self.cbm.get_production_coefficients() == np.array([0.7, 0.3, 0, 0])))

    def test_get_edges(self):
        actual = ['Stratosphere:60:0.7 --> Troposphere:50:0.3 : 0.5',
                  'Troposphere:50:0.3 --> Marine surface:900:0.0 : 0.2',
                  'Troposphere:50:0.3 --> Biosphere:1600:0.0 : 1']
        self.assertEqual(self.cbm.get_edges(), actual)

    def test_get_edge_objects(self):
        self.assertEqual(self.cbm.get_edges_objects(), self.edges)

    def test_get_nodes(self):
        actual = ['Stratosphere', 'Troposphere', 'Marine surface', 'Biosphere']
        self.assertEqual(self.cbm.get_nodes(), actual)

    def test_get_nodes_objects(self):
        self.assertEqual(self.cbm.get_nodes_objects(), self.nodes)

    def test_converted_fluxes_not_changed(self):
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
        self.assertTrue(np.all(cbm.get_converted_fluxes() == actual))

    def test_get_reservoir_contents(self):
        actual = np.array([60, 50, 900, 1600])
        self.assertTrue(jnp.all(self.cbm.get_reservoir_contents() == actual))

    def test_get_fluxes(self):
        actual = np.array([[0, 0.5, 0, 0], [0, 0, 0.2, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertTrue(jnp.all(self.cbm.get_fluxes() == actual))

    def test_converted_fluxes_changed(self):
        actual = np.array([[0, 35.008105, 0, 0], [35.008105, 0, 11.6693683, 58.34684167], [0, 11.6693683, 0, 0],
                           [0, 58.34684167, 0, 0]])
        self.assertTrue(jnp.all(jnp.allclose(self.cbm.get_converted_fluxes(), actual)))

    def test_add_node_not_box_class(self):
        cbm = CarbonBoxModel()
        try:
            cbm.add_nodes(['a'])
        except ValueError:
            pass
        else:
            self.fail('incorrect implementation. It accepted type string as a node')

    def test_add_node_duplicate(self):
        cbm = CarbonBoxModel()
        box1 = Box('t1', 101)
        cbm.add_nodes([box1, box1, box1])
        self.assertEqual(cbm.get_nodes(), [box1.get_name()])

    def test_add_edge_not_flow_class(self):
        cbm = CarbonBoxModel()
        try:
            cbm.add_edges(['a'])
        except ValueError:
            pass
        else:
            self.fail('incorrect implementation. It accepted type string as an edge')

    def test_add_edge_not_flow_class2(self):
        cbm = CarbonBoxModel()
        b1 = Box('t1', 101)
        try:
            cbm.add_edges([b1])
        except ValueError:
            pass
        else:
            self.fail('incorrect implementation. It accepted type Box as an edge')

