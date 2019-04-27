import numpy as np
from numpy import genfromtxt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preprocess_train(training_data):
    train_labels = training_data[1:, -1]
    train_labels = (train_labels == 's').astype(np.int32)

    eventIds = training_data[1:,0].astype(np.float32)
    weights = training_data[1:,31].astype(np.float32)
    train_data = training_data[1:, 1:31]
    train_data = train_data.astype(np.float32)

    a, b = np.where(train_data == -999)

    new_r, new_c = np.where(train_data != -999)
    mean_list = np.mean(train_data[new_r][new_c], axis=0)

    for i in range(len(a)):
        train_data[a[i]][train_data[a[i]] == -999] = mean_list[b[i]]
    x, y = np.where(train_data == -999)

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)

    pca = PCA(.95)
    pca.fit(train_data)
    # print pca.n_components_
    train_data = pca.transform(train_data)

    return train_data, train_labels, eventIds, weights


def main():
    training_data = genfromtxt('higgs-boson/training.csv', dtype=str, delimiter=',')
    # test_data = genfromtxt('higgs-boson/testing.csv', dtype=str, delimiter=',')

    # print training_data.shape #(250001,33)
    train_data, train_labels, eventIds, weights = preprocess_train(training_data)

    print train_data.shape, train_labels.shape, eventIds.shape, weights.shape
    #(250000, 30) (250000,) (250000,) (250000,)

    training_data = np.column_stack((eventIds, train_data, weights, train_labels))
    print training_data.shape  #(250000, 33)

    # np.savetxt("higgs-boson/preprocessed_training_data.csv", training_data, delimiter=",")
    np.savetxt("higgs-boson/preprocessed_training_data_pca.csv", training_data, delimiter=",")


if __name__== "__main__":
    main()
