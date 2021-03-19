from typing import List

from ML.P4.P4_neuron_laag import NeuronLaag


class NeuronNetwork:
    def __init__(self, lagen: List[NeuronLaag]):
        self.lagen = lagen
        self.output_neuronen = []
        self.losses = []

    def feed_forward(self, inputs: List[float]):
        """Met de feed-forward runnen we het hele netwerk en slaan we op wat de uiteindelijke antwoorden zijn van het
        netwerk."""
        alle_outputs = [inputs]
        for laag in self.lagen:  # Per laag

            x = laag.laag_forward(alle_outputs[-1])

            alle_outputs.append(x)  # Pak de output van de laatste laag in 'alle_outputs' en pak de laatste output als
                                    # input voor de volgende laag.

        self.output_neuronen = alle_outputs

    def loss(self, targets):
        """Hier berekenen we de loss van het netwerk met een bepaalde input."""
        loss_total = 0
        for i in range(len(targets)):
            loss_total += (targets[i] - self.output_neuronen[-1][i]) ** 2
        self.losses.append(loss_total / len(targets))

    def total_loss(self, inputs, targets):
        """Hier berekenen we de totale loss van het hele netwerk."""
        for i in range(len(inputs)):
            self.feed_forward(inputs[i])
            self.loss(targets[i])

        return sum(self.losses) / len(self.losses)

    def update(self, inputs: List[float], targets: List[float]):
        """In deze definitie geven we de inputs van een neuronlaag door naar de volgende laag tot we uiteindelijk
        de outputlaag bereiken."""
        self.feed_forward(inputs)


        for i in range(0, len(self.lagen[-1].neurons)):  # Bereken de errors en dergelijke van de outputlayer
            self.lagen[-1].neurons[i].output_neuron(targets[i])


        hidden_neurons = self.lagen[:-1]  # Alle hidden neuron lagen

        for i in range(len(hidden_neurons) - 1, -1, -1):  # Bereken de errors en dergelijke van de hidden layers
            for j in range(0, len(hidden_neurons[i].neurons)):
                neuron = hidden_neurons[i].neurons[j]
                neuron_index = self.lagen[i].neurons.index(neuron)
                weights = self.lagen[i + 1].get_weights(neuron_index)
                errors = self.lagen[i + 1].get_errors()
                self.lagen[i].neurons[j].hidden_neuron(weights, errors)

        for i in range(len(self.lagen) - 1, -1, -1):  # Hier updaten we het hele netwerk
            self.lagen[i].update_layer()

    def train(self, inputs: List[List[float]], targets: List[List[float]], epochs: int):
        """In deze functie gaan we het netwerk trainen. We geven het netwerk een bepaald aantal epochs om te runnen."""
        for _ in range(0, epochs):
            if self.total_loss(inputs, targets) < 0.01:
                break
            for i in range(0, len(inputs)):
                self.update(inputs[i], targets[i])
                self.losses = []

    def __str__(self):
        """Informatie van het neuron network"""
        neuronen = ''
        for i in range(0, len(self.lagen) - 1):
            neuronen += str(self.lagen[i].name) + ', '
        neuronen += self.lagen[-1].name

        return 'In dit network zitten de volgende lagen: {}'.format(neuronen)
