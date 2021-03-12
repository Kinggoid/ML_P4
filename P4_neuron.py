from typing import List

import numpy as np


class Neuron:
    error = None
    output = None
    derivative = None
    gradients = None
    delta_weights = None
    delta_bias = None
    target = None
    inputs = None

    def __init__(self, weights: list, bias: float, learn_rate: float, name: str):
        self.weights = weights
        self.name = name
        self.bias = bias
        self.learn_rate = learn_rate

    def calculate_output(self, inputs: List[float]):
        """In deze definitie kijken we bij een neuron welke inputs hij krijgt en hoe de weights, bias en de sigmoid funcite
        dit beïnvloeden. Dit antwoord geven we terug."""
        self.inputs = inputs
        inputs_met_weight = [self.weights[i] * inputs[i] for i in range(0, len(inputs))]
        update = sum(inputs_met_weight) + self.bias
        self.output = 1 / (1 + np.exp(-update))


    def N_derivative(self):
        """Bereken de derivative van deze neuron."""
        self.derivative = self.output * (1 - self.output)

    def output_neuron_error(self, target: float):
        """Bereken de error van deze output neuron."""
        self.target = target
        self.error = self.derivative * -(target - self.output)

    def hidden_neuron_error(self, weights: List[float], error: List[float]):
        """Bereken de error van deze hidden neuron"""
        average_error = 0
        for i in range(0, len(weights)):
            average_error += weights[i] * error[i]
        self.error = self.derivative * average_error

    def N_gradient(self):
        """Bereken de gradients van deze neuron en de neurons uit de vorige laag."""
        lst = []
        for i in self.inputs:
            lst.append(i * self.error)
        self.gradients = lst

    def N_delta(self):
        """Bereken de delta van de weights en van de bias van deze neuron."""
        lst = []
        for i in self.gradients:
            lst.append(self.learn_rate * i)
        self.delta_weights = lst
        self.delta_bias = self.learn_rate * self.error

    def update(self):
        """Hier updaten we de weights en de biases van deze neuron."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.delta_weights[i]
        self.bias -= self.delta_bias

    def output_neuron(self, target):
        """Hier voeren we alle nodige functies voor een output neuron uit."""
        self.N_derivative()
        self.output_neuron_error(target)
        self.N_gradient()
        self.N_delta()

    def hidden_neuron(self, weights, errors):
        """Hier voeren we alle nodige functies voor een hidden neuron uit."""
        self.N_derivative()
        self.hidden_neuron_error(weights, errors)
        self.N_gradient()
        self.N_delta()

    def __str__(self):
        """Informatie van de neuron"""
        return 'Mijn naam is {}. Mijn  input variabelen. Mijn bias is {}.'.format(self.name, str(len(self.weights)), str(self.bias))



