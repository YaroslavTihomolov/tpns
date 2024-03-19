from task2 import NeuronLink
from task2.Neuron import Neuron


class ActivationNeuron(Neuron):
    def __init__(self, links: [NeuronLink], activation_function):
        super().__init__(links)
        self.__activation_function = activation_function

    def run(self):
        value: float = self.__activation_function(self._sum)
        for link in self.get_links():
            neuron = link.get_neuron()
            neuron.add_data(value * link.get_weight())

    def get_activation_data(self):
        return self.__activation_function(self._sum)