import random
from enum import Enum, auto

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import math

from task2.ActivationNeuron import ActivationNeuron
from task2.Neuron import Neuron
from task2.NeuronLink import NeuronLink
from task2.Perceptron import Perceptron


class Brand(Enum):
    ACER = auto()
    LENOVO = auto()
    HP = auto()
    ASUS = auto()
    DELL = auto()


class EnumNotFoundException(Exception):
    pass


class EnumUtils:
    @staticmethod
    def parse(enum_type, name):
        try:
            return enum_type[name.upper()]
        except KeyError:
            raise EnumNotFoundException(enum_type, name)


def activation1(x: float) -> float:
    return sigmoid(x, 0.3)


def activation2(x: float) -> float:
    return sigmoid(x, 0.5)

def activation3(x: float) -> float:
    return sigmoid(x, 0.7)

def activation4(x: float) -> float:
    return sigmoid(x, -0.3)

def activation5(x: float) -> float:
    return sigmoid(x, 0.1)

def activation6(x: float) -> float:
    return sigmoid(x, -0.2)

def activation7(x: float) -> float:
    return sigmoid(x, 0.25)

def activation8(x: float) -> float:
    return sigmoid(x, 0.9)

def activation9(x: float) -> float:
    return sigmoid(x, -0.9)

def activation10(x: float) -> float:
    return sigmoid(x, -0.7)


def sigmoid(x: float, alpha: float) -> float:
    return 1 / (1 + math.exp(-alpha * x))


def fake(x: float):
    return x

funcs = [activation1, activation2, activation3, activation4, activation5, activation6, activation8, activation9, activation10]


def create_net() -> [[Neuron]]:
    output_neuron = ActivationNeuron(None, activation7)

    # output_link = NeuronLink(output_neuron, random.random())

    # neuron_2 = ActivationNeuron([output_link], activation)
    # neuron_2 = ActivationNeuron([output_link], activation)
    input_neurons = []
    weights = []
    layer_1_neurons = []
    layer_2_neurons = []
    count_2 = 2
    for h1 in range(count_2):
        neuron = ActivationNeuron([NeuronLink(output_neuron, random.uniform(-1, 1))], funcs[h1])
        layer_2_neurons.append(neuron)
    for h2 in range(5):
        links_to_2 = []
        for n2 in layer_2_neurons:
            links_to_2.append(NeuronLink(n2, random.uniform(-1, 1)))
        neuron = ActivationNeuron(links_to_2, funcs[h2 + count_2])
        layer_1_neurons.append(neuron)
    for ind in range(len(laptops.columns) - 2):
        # weight2 = random.random()
        # weights.append(weight2)
        links = []
        for n in layer_1_neurons:
            weight1 = random.uniform(-1, 1)
            weights.append(weight1)
            link_1 = NeuronLink(n, weight1)
            links.append(link_1)
        # link_2 = NeuronLink(neuron_2, weight2)
        neuron = ActivationNeuron(links, fake)
        input_neurons.append(neuron)
    print(weights)
    return [input_neurons, layer_1_neurons, layer_2_neurons, [output_neuron]]


scaler = MinMaxScaler()


def normalize_data(data):
    scaleddata = scaler.fit_transform(data)
    return pd.DataFrame(scaleddata, columns=data.columns)


def load_and_normalize_data(filename):
    data = pd.read_csv(filename)
    normalized_data = normalize_data(data)
    return normalized_data


laptops1 = pd.read_csv('Laptop_price2.csv')
max_value = laptops1['Price'].max()
min_value = laptops1['Price'].min()

laptops = load_and_normalize_data('Laptop_price2.csv')


def un_normalize(val: float):
    return (max_value - min_value) * val + min_value


def run(per: Perceptron, laptops, iter):
    for i in range(len(laptops.values)):
        max_e = 0
        rel = laptops1.values[i]
        per.fill(laptops.values[i])
        res = per.run()
        un_res = un_normalize(res)
        dif = abs(un_res - rel[-1])
        if dif > max_e:
            max_e = dif
        if iter == 999:
            print([un_normalize(res), un_normalize(laptops.values[i][-1]), abs(res - laptops.values[i][-1]),
               abs(un_res - rel[-1])])
        per.error(laptops.values[i][-1])
        per.study()
        per.clean()
        if i == 999:
            print(max_e)
    print("\\\\\\\\\\Education finish\\\\\\\\\\")


if __name__ == "__main__":
    p = Perceptron(create_net())
    for s in range(500):
        run(p, laptops, s)
