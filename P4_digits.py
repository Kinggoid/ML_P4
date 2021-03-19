import random
from typing import List

from sklearn.datasets import load_digits
from sklearn import preprocessing
import numpy as np
import pandas as pd
from ML.P4.P4_neuron import Neuron
from ML.P4.P4_neuron_laag import NeuronLaag
from ML.P4.P4_neuron_netwerk import NeuronNetwork
from sklearn.model_selection import train_test_split


def predict(data: List[List[float]], netwerk: NeuronNetwork):
    """Gegeven de data van een plant kunnen we met een bepaalde zekerheid voorspellen welke plant dit is. In deze
    functie veranderen we onze output. De plant die de hoogste kans heeft om correct te zijn, is de plant die wij
    gaan gokken."""
    antwoorden = []
    for i in data:
        netwerk.feed_forward(i)
        output = netwerk.output_neuronen[-1]  # Pak alleen de outputlaag
        index = output.index(max(output))  # Welke plant heeft de grootste kans om correct te zijn? Wanneer twee of
                                           # drie planten dezelfde accuracy hebben, wordt de eerste gepakt.
        lege_lijst = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        lege_lijst[index] = 1
        antwoorden.append(lege_lijst)
    return antwoorden


def accuracy(targets: List[List[float]], predict: List[List[float]]):
    """Check of de predictie van jouw netwerk accuraat is."""
    correct = 0
    for i in range(len(targets)):
        if predict[i] == targets[i]:
            correct += 1
    return correct / len(targets) * 100


def create_layer(learn_rate, neurons, amount_of_weights, which_layer):
    """Hier creëeren we een netwerk laag. Je geeft mee wat de learning rate is, hoeveel neuronen er in deze laag moeten
    komen te zitten, hoeveel neuronen er in de vorige laag zitten (om te bepalen hoeveel weights er moeten zijn) en
    welk laag dit is."""
    layer = []

    for neuron in range(0, neurons):
        sampl = np.random.uniform(low=-2, high=2, size=(amount_of_weights + 1,))  # Maak een aantal random getallen aan.
        layer.append(Neuron(sampl[0:-1], sampl[-1], learn_rate, 'N' + str(neuron) + '_' + str(which_layer)))

    return NeuronLaag(layer, 'laag' + str(which_layer))


def create_network(layers):
    """Hier maken we een neuron network aan."""
    return NeuronNetwork(layers)


def main():
    """Main functie."""
    digits = load_digits()

    data = preprocessing.normalize(digits.data)  # Normaliseer de dataset om alle inputs tussen de 0 en de 1 te krijgen.

    targets = []
    for i in digits.target:
        normal_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        normal_list[i] = 1
        targets.append(normal_list)

    shuffle = list(zip(data, targets))  # Schud de data en de targets op dezelfde wijze. Anders krijg je heel vaak
                                        # dezelfde planten achter elkaar en dan duurt het langer om het netwerk te trainen.
    random.shuffle(shuffle)
    data, targets = zip(*shuffle)

    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.20, random_state=1)

    L1 = create_layer(1, 10, 64, 1)
    L2 = create_layer(1, 10, 10, 2)
    L3 = create_layer(1, 10, 10, 3)
    network = create_network([L1, L2, L3])

    network.train(data, targets, 1000)

    print('De trainset heeft een nauwkeurigheid van ' + str(accuracy(y_train, predict(X_train, network))) + '%')
    print('De testset heeft een nauwkeurigheid van ' + str(accuracy(y_test, predict(X_test, network))) + '%')


main()
