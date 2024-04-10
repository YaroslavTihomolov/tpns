from numpy import ndarray

from task3 import ActivationNeuron
from task3.Layer import Layer


class Perceptron:
    def __init__(self, layers: [Layer]):
        self.layers: [[ActivationNeuron]] = layers
        self.__education_speed = 0.5

    def run(self) -> float:
        for layer in self.layers[:-1]:
            for neuron in layer.get_neurons():
                neuron.run()
        return self.layers[-1].get_neurons()[-1].get_activation_data()


    @staticmethod
    def __calc_output_error(output_value: float, target_value: float) -> float:
        return (output_value - target_value) * output_value * (1 - output_value)

    @staticmethod
    def __calc_error(actual_value: float, next_error: float) -> float:
        return next_error * actual_value * (1 - actual_value)

    @staticmethod
    def count_next_error(neuron: ActivationNeuron):
        error = 0
        for link in neuron.get_links():
            next_neuron = link.get_neuron()
            error += next_neuron.get_error() * link.get_weight()
        return error

    def error(self, target_value: float):
        output_neuron = self.layers[-1].get_neurons()[-1]
        output_error: float = self.__calc_output_error(output_neuron.get_activation_data(), target_value)
        output_neuron.set_error(output_error)
        for layer in reversed(self.layers[1:-1]):
            for neuron in layer.get_neurons():
                next_error = self.count_next_error(neuron)
                error = self.__calc_error(neuron.get_activation_data(), next_error)
                neuron.set_error(error)

    def __update_weight(self, neuron: ActivationNeuron):
        for link in neuron.get_links():
            next_neuron: ActivationNeuron = link.get_neuron()
            dif_weight = -self.__education_speed * next_neuron.get_error() * neuron.get_activation_data()
            link.sum_weight(dif_weight)

    def study(self):
        for layer in reversed(self.layers[:-1]):
            for neuron in layer.get_neurons():
                self.__update_weight(neuron)

    def fill(self, values: ndarray):
        for i, inp_neuron in enumerate(self.layers[0].get_neurons()):
            inp_neuron.set_data(values[i])

    def clean(self):
        for layer in self.layers[1:]:
            for neuron in layer.get_neurons():
                neuron.set_data(0)

    def step(self, row):
        self.fill(row)
        self.run()
        self.error(row[-1])
        self.study()
        self.clean()


