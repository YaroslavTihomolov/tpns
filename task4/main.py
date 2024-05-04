# import random
#
# import numpy as np
# import pandas as pd
#
#
#
# def normalize_and_save(file_name: str):
#     df = pd.read_csv(file_name)
#     df.iloc[:, 0] /= 9
#     df.iloc[:, 1:] /= 255
#     for i in range(1, 29):
#         index = str(i)
#         new_columns = [index + 'x29', index + 'x30', index + 'x31', index + 'x32']
#         insert_position = df.columns.get_loc(index + 'x28') + 1
#         for column in new_columns[::-1]:
#             df.insert(insert_position, column, 0)
#     for i in range(29, 33):
#         for j in range(1, 33):
#             df.insert(len(df.columns), str(i) + 'x' + str(j), 0)
#     df.to_csv('normalize_' + file_name, index=False)
#
# def get_matrix(full_matrix: [float], str_number: int, column_num: int, size: int):
#     return [full_matrix[index * 32 + column_num:index * 32 + size + column_num] for index in range(str_number, str_number + size)]
#
# def mult_matr(part_matrix: [float], mult_matrix: [float]) -> float:
#     value = 0
#     for i in range(5):
#         for j in range(5):
#             value += part_matrix[i][j] * mult_matrix[i][j]
#     return value
#
# def max_pulling(matrix_to_pull: [float]) -> float:
#     return np.max(matrix_to_pull)
#
# if __name__ == "__main__":
#     df = pd.read_csv('normalize_mnist_train.csv')
#     s = df.values[0][1:]
#     filter = [[random.randint(-1, 1) for _ in range(5)] for _ in range(5)]
#     new_matrix = [float]
#     next_matrix = []
#     for i in range(28):
#         for j in range(28):
#             part_matrix = get_matrix(s, i, j, 5)
#             new_value = mult_matr(part_matrix, filter)
#             next_matrix.append(new_value)
#     matrix_after_pulling = []
#     for i in range(0, 27, 2):
#         for j in range(0, 27, 2):
#             part_matrix = get_matrix(s, i, j, 2)
#             new_value = max_pulling(part_matrix)
#             matrix_after_pulling.append(new_value)
#     for i in range(784):
#         if i % 28 == 0:
#             print('\n')
#         else:
#             print("{:.1f}".format(next_matrix[i]), end = ' ')
#     for i in range(196):
#         if i % 14 == 0:
#             print('\n')
#         else:
#             print("{:.1f}".format(matrix_after_pulling[i]), end = ' ')
#
#
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

from task4.Dense import Dense
from task4.Reshape import Reshape
from task4.Sigmoid import Sigmoid
from task4.convolutional import Convolutional
from task4.losses import binary_cross_entropy, binary_cross_entropy_prime
from task4.metrics import accuracy, precision, recall, auc_roc
from task4.network import train, predict


def to_categorical(y, num_classes=None):
    if num_classes is None:
        num_classes = np.max(y) + 1

    one_hot = np.zeros((len(y), num_classes))
    for i, label in enumerate(y):
        one_hot[i, label] = 1

    return one_hot


def preprocess_data(x, y, limit):
    all_indices = np.array([], dtype=int)

    for label in range(10):
        indices = np.where(y == label)[0][:limit]
        all_indices = np.concatenate((all_indices, indices))

    all_indices = np.array(all_indices, dtype=int)
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 10, 1)
    return x, y

#     for i in range(10):
#         indices = np.where(y == i)[0][:limit]

# load MNIST from server
df_test = pd.read_csv('mnist_test.csv')
df_train = pd.read_csv('mnist_train.csv')
(x_train, y_train), (x_test, y_test) = [[df_train.iloc[:, 1:].values, df_train.iloc[:, 0].values.flatten()],
                                        [df_test.iloc[:, 1:].values, df_test.iloc[:, 0].values.flatten()]]
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]

train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

predicted_labels = [int]
true_labels = [int]

# def plot_roc_curve(y_true, y_score):
#     # Вычисление значений ROC кривой и ее площади AUC
#     label_encoder = LabelEncoder()
#     y_true_binary = label_encoder.fit_transform(y_true)
#     fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
#     roc_auc = auc(fpr, tpr)
#
#     # Построение ROC кривой
#     plt.figure()
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.show()

predicted_label = None

for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
    predicted_label = np.argmax(output)
    true_label = np.argmax(y)
    predicted_labels.append(predicted_label)
    true_labels.append(true_label)

for i in range(10):
    print(str(i) + ":")
    print("     acсuracy:  " + str(accuracy(predicted_labels, true_labels, i)))
    print("     precision: " + str(precision(predicted_labels, true_labels, i)))
    print("     recall: " + str(recall(predicted_labels, true_labels, i)))
