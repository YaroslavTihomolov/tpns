from task2 import NeuronLink


class Neuron:

    def __init__(self, links: [NeuronLink]):
        self.__links = links
        self._sum: float = 0
        self.__error: float = 0

    def add_data(self, value: float):
        self._sum += value

    def set_data(self, value: float):
        self._sum = value

    def get_data(self):
        return self._sum

    def get_links(self) -> [NeuronLink]:
        return self.__links

    def get_error(self):
        return self.__error

    def set_error(self, error: float):
        self.__error = error
