from typing import List

from ML.P4.P4_neuron import Neuron


class NeuronLaag:
    def __init__(self, neurons: List[Neuron], name: str):
        self.neurons = neurons
        self.name = name

    def laag_forward(self, inputs: List[int]):
        """Bereken van elke neuron, in deze laag, zijn output en geef alle outputs terug."""
        outputs = []
        for neuron in self.neurons:
            neuron.calculate_output(inputs)
            outputs.append(neuron.output)
        return outputs

    def get_weights(self, which_neuron):
        """Haal van elk van de neuronen uit deze laag hun weight die zij hebben met een gegeven neuron uit een vorige
        laag."""
        weights = []
        for i in self.neurons:
            weights.append(i.weights[which_neuron])
        return weights

    def get_errors(self):
        """Haal van elk van de neuronen uit deze laag hun error."""
        errors = []
        for i in self.neurons:
            errors.append(i.error)
        return errors

    def __str__(self):
        """Informatie van de neuronlaag."""
        neuronen = ''
        for i in range(0, len(self.neurons) - 1):
            neuronen += str(self.neurons[i].name) + ', '
        neuronen += self.neurons[-1].name

        return 'Ik ben de neuronlaag {} en ik bevat de volgende neuronen: {}'.format(self.name, neuronen)

