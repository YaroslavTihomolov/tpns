from task3.ActivationNeuron import ActivationNeuron


class Layer:

    def __init__(self, neurons: [ActivationNeuron]):
        self.__neurons = neurons
        self.__sum_res = 0

    def get_neurons(self):
        return self.__neurons

    def run(self):
        for neuron in self.__neurons:
            neuron.run()

