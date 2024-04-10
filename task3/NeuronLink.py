from task3 import Neuron


class NeuronLink:

    def __init__(self, neuron: Neuron, weight: float):
        self.__neuron = neuron
        self.__weight = weight

    def get_weight(self):
        return self.__weight

    def get_neuron(self):
        return self.__neuron

    def set_weight(self, weight: float):
        self.__weight = weight

    def sum_weight(self, weight: float):
        self.__weight += weight