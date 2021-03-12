import math
import unittest

from ML.P4.P4_neuron import Neuron
from ML.P4.P4_neuron_laag import NeuronLaag
from ML.P4.P4_neuron_netwerk import NeuronNetwork


class MyTestCase(unittest.TestCase):
    def test_AND(self):
        inputs = [[0, 0], [1, 0], [0, 1], [1, 1]]
        verwachte_outputs = [[0], [0], [0], [1]]
        epochs = 100

        AND = Neuron([-0.5, 0.5], 1.5, 1, 'AND gate')
        laag_een = NeuronLaag([AND], 'L1')
        ANDer = NeuronNetwork([laag_een])

        for i in range(0, len(inputs)):
            ANDer.feed_forward(inputs[i])
            ANDer.loss(verwachte_outputs[i])

        ANDer.train(inputs, verwachte_outputs, epochs)

        antwoorden = []

        for i in inputs:
            ANDer.feed_forward(i)
            tussen_antwoorden = []
            for j in ANDer.output_neuronen:
                if (j % 1) == 0.5:
                    tussen_antwoorden.append(int(math.ceil(j)))
                else:
                    tussen_antwoorden.append(int(round(j)))

            antwoorden.append(tussen_antwoorden)

        self.assertEqual(antwoorden, verwachte_outputs)  # Kijk of de outputs goed zijn

    def test_XOR(self):
        x1_1 = Neuron([0.2, -0.4], 0, 1, 'F')
        x1_2 = Neuron([0.7, 0.1], 0, 1, 'G')

        laag_een = NeuronLaag([x1_1, x1_2], 'L1')

        x2_1 = Neuron([0.6, 0.9], 0, 1, 'O')

        laag_twee = NeuronLaag([x2_1], 'L2')

        XOR = NeuronNetwork([laag_een, laag_twee])

        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        verwachte_outputs = [[0], [1], [1], [0]]
        epochs = 1000

        XOR.train(inputs, verwachte_outputs, epochs)

        antwoorden = []
        for i in inputs:
            XOR.feed_forward(i)
            tussen_antwoorden = []
            for j in XOR.output_neuronen:
                if (j % 1) == 0.5:
                    tussen_antwoorden.append(int(math.ceil(j)))
                else:
                    tussen_antwoorden.append(int(round(j)))
            antwoorden.append(tussen_antwoorden)

        self.assertEqual(antwoorden, verwachte_outputs)  # Kijk of de outputs goed zijn

    def test_Half_adder(self):
        x1_1 = Neuron([0.0, 0.1], 0, 1, 'F')
        x1_2 = Neuron([0.2, 0.3], 0, 1, 'G')
        x1_3 = Neuron([0.4, 0.5], 0, 1, 'H')

        laag_een = NeuronLaag([x1_1, x1_2, x1_3], 'L1')

        x2_1 = Neuron([0.6, 0.7, 0.8], 0, 1, 'S')
        x2_2 = Neuron([0.9, 1, 1.1], 0, 1, 'C')

        laag_twee = NeuronLaag([x2_1, x2_2], 'L2')

        Half_adder = NeuronNetwork([laag_een, laag_twee])

        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        verwachte_outputs = [[0, 0], [1, 0], [1, 0], [0, 1]]
        epochs = 1000

        Half_adder.train(inputs, verwachte_outputs, epochs)

        antwoorden = []
        for i in inputs:
            Half_adder.feed_forward(i)
            tussen_antwoorden = []
            for j in Half_adder.output_neuronen:
                if (j % 1) == 0.5:
                    tussen_antwoorden.append(int(math.ceil(j)))
                else:
                    tussen_antwoorden.append(int(round(j)))
            antwoorden.append(tussen_antwoorden)

        self.assertEqual(antwoorden, verwachte_outputs)  # Kijk of de outputs goed zijn


if __name__ == '__main__':
    unittest.main()
