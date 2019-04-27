import numpy as np
from numpy import genfromtxt
import math
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import neural_network


def process_data(t_data):
    labels = t_data[1:, -1]
    labels = (labels == 's').astype(np.int32)
    weights = t_data[1:, 31].astype(np.float32)
    data = t_data[1:, 1:31]
    data = data.astype(np.float32)
    return data, labels, weights


def normalize_data(data):
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data


def replace_missing_values(data):
    a, b = np.where(data == -999)
    new_r, new_c = np.where(data != -999)
    mean_list = np.mean(data[new_r][new_c], axis=0)
    for i in range(len(a)):
        data[a[i]][data[a[i]] == -999] = mean_list[b[i]]
    x, y = np.where(data == -999)
    print(x, y)
    return data


def run_pca(data):
    pca = PCA(n_components=25)
    pca.fit(data)
    data = pca.transform(data)
    return data


def calc_ams(w, y, p):
    w = np.asarray(w)
    y = np.asarray(y)
    p = np.asarray(p)
    s = np.sum(w * (y == 1) * (p == 1))
    b = np.sum(w * (y == 0) * (p == 1))
    b_r = 10.0
    radicand = 2 * ((s + b + b_r) * math.log(1.0 + s / (b + b_r)) - s)
    return math.sqrt(radicand)


def precision(labels, pred):
    precision = precision_score(labels, pred)
    return precision


def recall(labels, pred):
    recall = recall_score(labels, pred)
    return recall


def process_data_jet_num(train_data, train_labels, num, weights):
    if num == 0:
        data = train_data[train_data[:, 22] == 0]
        data = np.delete(data, [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28], 1)
        labels = train_labels[np.where(train_data[:, 22] == 0)]
        weights = weights[np.where(train_data[:, 22] == 0)]
    elif num == 1:
        data = train_data[train_data[:, 22] == 1]
        data = np.delete(data, [4, 5, 6, 12, 26, 27, 28, 22], 1)
        labels = train_labels[np.where(train_data[:, 22] == 1)]
        weights = weights[np.where(train_data[:, 22] == 1)]
    else:
        data_2 = train_data[train_data[:, 22] == 2]
        data_2 = np.delete(data_2, [22], 1)
        labels_2 = train_labels[np.where(train_data[:, 22] == 2)]
        labels_2_weights = weights[np.where(train_data[:, 22] == 2)]

        data_3 = train_data[train_data[:, 22] == 3]
        data_3 = np.delete(data_3, [22], 1)
        labels_3 = train_labels[np.where(train_data[:, 22] == 3)]
        labels_3_weights = weights[np.where(train_data[:, 22] == 3)]

        data = np.concatenate((data_2, data_3), axis=0)
        labels = np.concatenate((labels_2, labels_3), axis=0)
        weights = np.concatenate((labels_2_weights, labels_3_weights), axis=0)
    return data, labels, weights


def merge_jet_num(preds_0, preds_1, preds_2_3, test_0_labels, test_1_labels, test_2_3_labels, test_0_weights,
                  test_1_weights, test_2_3_weights):
    preds = np.concatenate((preds_0, preds_1, preds_2_3))
    labels = np.concatenate((test_0_labels, test_1_labels, test_2_3_labels))
    weights = np.concatenate((test_0_weights, test_1_weights, test_2_3_weights))
    print(preds.shape)
    print(labels.shape)
    print(weights.shape)
    print("jet-num-merged: ")
    print("Accuracy: ", accuracy_score(labels, preds))
    print("Precision: ", precision(labels, preds))
    print("Recall: ", recall(labels, preds))
    print("AMS score: ", calc_ams(weights.astype(np.float32), labels, preds))



if __name__ == "__main__":
    training_data = genfromtxt('higgs-boson/training.csv', dtype=str, delimiter=',')
    testing_data = genfromtxt('higgs-boson/testing.csv', dtype=str, delimiter=',')

    train_data, train_labels, train_weights = process_data(training_data)
    test_data, test_labels, test_weights = process_data(testing_data)

    data_jet_num_0, data_jet_num_0_labels, data_jet_num_0_weights = process_data_jet_num(train_data, train_labels, 0,
                                                                                         train_weights)
    data_jet_num_0 = normalize_data(data_jet_num_0)

    data_jet_num_1, data_jet_num_1_labels, data_jet_num_1_weights = process_data_jet_num(train_data, train_labels, 1,
                                                                                         train_weights)
    data_jet_num_1 = normalize_data(data_jet_num_1)

    data_jet_num_2_3, data_jet_num_2_3_labels, data_jet_num_2_3_weights = process_data_jet_num(train_data, train_labels,
                                                                                               2, train_weights)
    data_jet_num_2_3 = normalize_data(data_jet_num_2_3)

    data_jet_num_0_test, data_jet_num_0_test_labels, data_jet_num_0_test_weights = process_data_jet_num(test_data,
                                                                                                        test_labels, 0,
                                                                                                        test_weights)
    data_jet_num_0_test = normalize_data(data_jet_num_0_test)

    data_jet_num_1_test, data_jet_num_1_test_labels, data_jet_num_1_test_weights = process_data_jet_num(test_data,
                                                                                                        test_labels, 1,
                                                                                                        test_weights)
    data_jet_num_1_test = normalize_data(data_jet_num_1_test)

    data_jet_num_2_3_test, data_jet_num_2_3_test_labels, data_jet_num_2_3_test_weights = process_data_jet_num(test_data,
                                                                                                              test_labels,
                                                                                                              2,
                                                                                                              test_weights)
    data_jet_num_2_3_test = normalize_data(data_jet_num_2_3_test)

    train_data_0, train_labels_0, test_data_0, test_labels_0, weights_0 = data_jet_num_0,data_jet_num_0_labels,data_jet_num_0_test,data_jet_num_0_test_labels,data_jet_num_0_test_weights
    train_data_1, train_labels_1, test_data_1, test_labels_1, weights_1 = data_jet_num_1,data_jet_num_1_labels,data_jet_num_1_test,data_jet_num_1_test_labels,data_jet_num_1_test_weights
    train_data_23, train_labels_23, test_data_23, test_labels_23, weights_23 = data_jet_num_2_3,data_jet_num_2_3_labels,data_jet_num_2_3_test,data_jet_num_2_3_test_labels,data_jet_num_2_3_test_weights


    train_labels_0 = train_labels_0.reshape((train_labels_0.shape[0],1))
    test_labels_0 = test_labels_0.reshape((test_labels_0.shape[0], 1))
    train_labels_1 = train_labels_1.reshape((train_labels_1.shape[0], 1))
    test_labels_1 = test_labels_1.reshape((test_labels_1.shape[0], 1))
    train_labels_23 = train_labels_23.reshape((train_labels_23.shape[0], 1))
    test_labels_23 = test_labels_23.reshape((test_labels_23.shape[0], 1))
    weights_0 = weights_0.reshape((weights_0.shape[0], 1))
    weights_1 = weights_1.reshape((weights_1.shape[0], 1))
    weights_23 = weights_23.reshape((weights_23.shape[0], 1))


    print train_data_0.shape, train_labels_0.shape, test_data_0.shape, test_labels_0.shape, weights_0.shape
    print train_data_1.shape, train_labels_1.shape, test_data_1.shape, test_labels_1.shape, weights_1.shape
    print train_data_23.shape, train_labels_23.shape, test_data_23.shape, test_labels_23.shape, weights_23.shape

    preds_0 = neural_network.trainNN(train_data_0, train_labels_0, test_data_0, test_labels_0, weights_0)
    print("AMS score: ", calc_ams(weights_0.astype(np.float32), test_labels_0, preds_0))

    preds_1 = neural_network.trainNN(train_data_1, train_labels_1, test_data_1, test_labels_1, weights_1)
    print("AMS score: ", calc_ams(weights_1.astype(np.float32), test_labels_1, preds_1))

    preds_23 = neural_network.trainNN(train_data_23, train_labels_23, test_data_23, test_labels_23, weights_23)
    print("AMS score: ", calc_ams(weights_23.astype(np.float32), test_labels_23, preds_23))

    merge_jet_num(preds_0, preds_1, preds_23, test_labels_0, test_labels_1, test_labels_23, weights_0, weights_1, weights_23)

