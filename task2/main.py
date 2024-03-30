import random
from enum import Enum, auto

import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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


funcs = [activation1, activation2, activation3, activation4, activation5, activation6, activation8, activation9,
         activation10]

weights = [0.8874952000569718, -0.010798291486668221, -0.895671233591941, 0.8991818280074149, -0.14829317278140874,
           -0.09824399798373351, 0.06066835252968272, -0.6458409980982449, 0.9757270606330346, -0.48799259413662144,
           0.22283661184620085, 0.4720266180815522, 0.6781399991907722, -0.9049180146576987, -0.9416740138516686,
           -0.1298884249382024, -0.6947589329025334, -0.27546675298067713, 0.14885812269809962, -0.7779448462156644,
           -0.2918067782649145, -0.9206170229566881, 0.01856659491569257, -0.40814111052613145, -0.7616258437620247,
           0.9135118525099974, -0.339701160600435, -0.780731677117589, -0.8523993646918595, -0.4971144788539521,
           -0.37053080229811064, -0.2355260937321526, -0.10872013193325447, -0.38524195259491734, -0.8478320399438075,
           -0.5930826399514395, 0.8954270635075681, 0.3848955912066736, -0.4259334879994099, -0.5773529012486374,
           0.43776435500458133, -0.9619502830632931, 0.33290565763369706, -0.8066227800487129, -0.059773108421136145,
           -0.3207711342199533, -0.5134274749888967, -0.9334235151939734, 0.5740914892335829, -0.7401144838377487,
           0.8033058774284025, 0.2843378116225874, -0.27550814470652796, 0.42965095222365535, 0.9301013878836348,
           -0.9039569407780945, 0.9593801592112556, 0.2955845156577579, 0.06695271488396548, 0.8901808280449681,
           0.20419122321903016, 0.44257664381707484, -0.3119719987304572, -0.25733749436067854, -0.6129919698297732,
           0.14334925698970347, -0.6923613116271394]


def create_net(neuron_counts: [int]) -> [[Neuron]]:
    index_w = 0
    output_neuron = ActivationNeuron(None, activation7)
    neurons = [[output_neuron]]
    cur_layer = [output_neuron]
    for index, count in enumerate(reversed(neuron_counts)):
        next_layer = cur_layer.copy()
        cur_layer = []
        for i in range(count):
            links_to_next_layer = []
            for n in next_layer:
                links_to_next_layer.append(NeuronLink(n, weights[index_w]))
                index_w += 1
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
mushrooms1 = pd.read_csv('update_mushrooms.csv')
max_laptop_value = laptops1['Price'].max()
min_laptop_value = laptops1['Price'].min()


def foo(x):
    if not isinstance(x, str) or len(x) != 1:
        return None
    else:
        return ord(x)


# encoded_mushrooms = mushrooms.applymap(lambda x: foo(x))
# encoded_mushrooms.to_csv('encoded_mushrooms.csv', index=False)
# print(mushrooms.values)


def un_normalize(val: float, max_mushroom_value: float, min_mushroom_value: float) -> float:
    return (max_mushroom_value - min_mushroom_value) * val + min_mushroom_value


def categorize_mushroom(val: float):
    dist_to_p = abs(val - 112)
    dist_to_e = abs(val - 101)
    if dist_to_p < dist_to_e:
        return 112
    return 101


def run(per: Perceptron, data_set, iter, original_data_set):
    max_e = 0
    correct = 0
    for i in range(len(data_set.values)):
        rel = original_data_set.values[i]
        per.fill(data_set.values[i])
        res = per.run()
        un_res = un_normalize(res)
        print([un_res, categorize_mushroom(un_res), rel[-1]])
        if categorize_mushroom(un_res) == rel[-1]:
            correct += 1
        dif = abs(un_res - rel[-1])
        if dif > max_e:
            max_e = dif
        # print([un_normalize(res), un_normalize(data_set.values[i][-1]), abs(res - data_set.values[i][-1]),abs(un_res - rel[-1])])
        per.error(data_set.values[i][-1])
        per.study()
        per.clean()
    print(max_e)
    print('correct: ' + str(correct) + ' incorrect: ' + str(len(data_set.values) - correct) + '\n')
    print("\\\\\\\\\\Education finish\\\\\\\\\\")


def educate(perceptron: Perceptron, data_set):
    for i in range(len(data_set.values)):
        perceptron.step(data_set.values[i])
    print("\\\\\\\\\\Education finish\\\\\\\\\\")

def count_accuracy(true_positive, true_negative, false_positive, false_negative) -> float:
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

def precision(true_positive, false_positive) -> float:
    return true_positive / (true_positive + false_positive)

def recall(true_positive, false_negative) -> float:
    return true_positive / (true_positive + false_negative)

def count_metrics(true_positive, true_negative, false_positive, false_negative):
    print("accuracy: " + str(count_accuracy(true_positive, true_negative, false_positive, false_negative)))
    print("precision: " + str(precision(true_positive, false_positive)))
    print("recall: " + str(precision(true_positive, false_negative)))

def check(perceptron: Perceptron, data_set, original_data_set):
    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0
    max_mushroom_value = original_data_set['poisonous'].max()
    min_mushroom_value = original_data_set['poisonous'].min()
    for i in range(len(original_data_set.values)):
        perceptron.fill(data_set.values[i])
        val = perceptron.run()
        perceptron.clean()
        actual_value = categorize_mushroom(un_normalize(val, max_mushroom_value, min_mushroom_value))
        expected_value = original_data_set.values[i][-1]
        if actual_value == expected_value == 112:
            true_positive += 1
        if actual_value == expected_value == 101:
            true_negative += 1
        if expected_value == 112 and actual_value != expected_value:
            false_negative += 1
        if expected_value == 101 and actual_value != expected_value:
            false_negative += 1
    count_metrics(true_positive, true_negative, false_positive, false_negative)


def educate_and_check(data_set, original_data_set, section_count):
    split_data = np.array_split(data_set, section_count)
    split_data_original = np.array_split(original_data_set, section_count)
    for i in range(section_count):
        perceptron = Perceptron(create_net([len(mushrooms.columns) - 1, 4, 3]))
        data = []
        for j in range(section_count):
            if j != i:
                data.append(split_data[j])
        merged_data = pd.concat(data)
        educate(perceptron, merged_data)
        check(perceptron, split_data[i], split_data_original[i])


if __name__ == "__main__":
    laptops = load_and_normalize_data('Laptop_price2.csv')
    mushrooms = load_and_normalize_data('update_mushrooms.csv')
    perceptron = Perceptron(create_net([len(mushrooms.columns) - 1, 4, 3]))
    # run(perceptron, mushrooms, 1, mushrooms1)
    educate_and_check(mushrooms, mushrooms1, 4)
