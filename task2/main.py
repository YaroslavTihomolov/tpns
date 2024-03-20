import random
from enum import Enum, auto
from ucimlrepo import fetch_ucirepo

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

def create_net(neuron_counts: [int]) -> [[Neuron]]:
    output_neuron = ActivationNeuron(None, activation7)
    neurons = [[output_neuron]]
    cur_layer = [output_neuron]
    for index, count in enumerate(reversed(neuron_counts)):
        next_layer = cur_layer.copy()
        cur_layer = []
        for i in range(count):
            links_to_next_layer = []
            for n in next_layer:
                links_to_next_layer.append(NeuronLink(n, random.uniform(-1, 1)))
            func = fake if index == len(neuron_counts) - 1 else funcs[i]
            cur_layer.append(ActivationNeuron(links_to_next_layer, func))
        neurons.append(cur_layer)
    return neurons[::-1]


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
    mushroom = fetch_ucirepo(id=73)

    # data (as pandas dataframes)
    X = mushroom.data.features
    y = mushroom.data.targets

    # metadata
    print(mushroom.data)

    # variable information
    # print(mushroom.variables)
    # p = Perceptron(create_net([6, 4, 2]))
    # for s in range(500):
    #     run(p, laptops, s)
