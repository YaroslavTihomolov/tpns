from numpy import ndarray

from task2 import ActivationNeuron


class Perceptron:
    def __init__(self, neurons: [[ActivationNeuron]]):
        self.neurons: [[ActivationNeuron]] = neurons
        self.__education_speed = 1

    def run(self) -> float:
        for layer in self.neurons[:-1]:
            for neuron in layer:
                neuron.run()
        return self.neurons[-1][-1].get_activation_data()


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
        output_neuron = self.neurons[-1][-1]
        output_error: float = self.__calc_output_error(output_neuron.get_activation_data(), target_value)
        output_neuron.set_error(output_error)
        for layer in reversed(self.neurons[1:-1]):
            for neuron in layer:
                next_error = self.count_next_error(neuron)
                error = self.__calc_error(neuron.get_activation_data(), next_error)
                neuron.set_error(error)

    def __update_weight(self, neuron: ActivationNeuron):
        for link in neuron.get_links():
            next_neuron: ActivationNeuron = link.get_neuron()
            dif_weight = -self.__education_speed * next_neuron.get_error() * neuron.get_activation_data()
            link.sum_weight(dif_weight)

    def study(self):
        for layer in reversed(self.neurons[:-1]):
            for neuron in layer:
                self.__update_weight(neuron)

    def fill(self, values: ndarray):
        i = 1
        for inp_neuron in self.neurons[0]:
            inp_neuron.set_data(values[i])
            i += 1

    def clean(self):
        for layer in self.neurons[1:]:
            for neuron in layer:
                neuron.set_data(0)




