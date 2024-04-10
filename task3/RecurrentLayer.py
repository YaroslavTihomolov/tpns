from task3.Layer import Layer


class RecurrentLayer:

    def __init__(self, input_layer: Layer, output_layer: Layer, b_h: Layer):
        self.__input_layer = input_layer
        self.__output_layer = output_layer
        self.__b_h = b_h

    def run(self):
        self.__input_layer.run()

