import numpy as np


def accuracy(predicted_labels: [int], true_labels: [int], val: int) -> float:
    tp = tn = fp = fn = 0

    for predicted, true in zip(predicted_labels, true_labels):
        if predicted == val and true == val:
            tp += 1
        elif predicted != val and true != val:
            tn += 1
        elif predicted == val and true != val:
            fp += 1
        elif predicted != val and true == val:
            fn += 1

    return (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0


def precision(predicted_labels: [int], true_labels: [int], val: int) -> float:
    count_true_positive: int = 0
    count_false_positive: int = 0

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == true_labels[i] == val:
            count_true_positive += 1
        elif predicted_labels[i] == val or true_labels[i] == val:
            count_false_positive += 1

    return count_true_positive / (count_true_positive + count_false_positive)


def auc_roc(true_labels, predicted_probs):
    sorted_indices = np.argsort(predicted_probs)[::-1]
    sorted_labels = true_labels[sorted_indices]

    tpr = fpr = auc = 0

    for label in sorted_labels:
        if label == 1:
            tpr += 1
        else:
            fpr += 1
            auc += tpr

    auc /= (tpr * fpr)

    return auc


def recall(predicted_labels: [int], true_labels: [int], val: int) -> float:
    count_true_predicted: int = 0
    false_negative = 0

    for i in range(len(predicted_labels)):
        if predicted_labels[i] == true_labels[i] == val:
            count_true_predicted += 1
        elif true_labels[i] == val:
            false_negative += 1

    return count_true_predicted / (count_true_predicted + false_negative)