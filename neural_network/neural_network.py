import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,recall_score
import math

# n_input = 30 * 1
n_classes = 2
batch_size = 32
n_epoch = 10
learning_rate = 0.002


def compute_s_b(w, y, p):
    w = np.asarray(w)
    y = np.asarray(y)
    p = np.asarray(p)
    s = np.sum(w * (y == 1) * (p == 1))
    b = np.sum(w * (y == 0) * (p == 1))
    return s, b


def ams(s, b, b_r=10):                  # b_r: constant regularization term
    radicand = 2 * ((s + b + b_r) * math.log(1.0 + s / (b + b_r)) - s)
    if radicand < 0:
        print 'radicand is negative. Exiting'
        exit()
    else:
        return math.sqrt(radicand)


def neural_net(features):
    hidden_layer_1 = fully_connected(features, 128, activation_fn=tf.nn.sigmoid)
    hidden_layer_2 = fully_connected(hidden_layer_1, 64, activation_fn=tf.nn.relu)
    hidden_layer_3 = fully_connected(hidden_layer_2, 32, activation_fn=tf.nn.sigmoid)
    output_layer = fully_connected(hidden_layer_3, 1, activation_fn=tf.nn.sigmoid)
    return output_layer


def trainNN(trainX, trainY, testX, testY, test_weights):
    n_input = trainX.shape[1] * 1
    features = tf.placeholder(dtype=tf.float32, shape=[None, n_input])
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    logits = neural_net(features)
    loss = tf.losses.log_loss(labels=labels, predictions=logits)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    train_step = optimizer.minimize(loss)

    preds = tf.math.round(logits)
    accuracy = tf.metrics.accuracy(labels, preds)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            n_batches = trainX.shape[0]//batch_size
            for iteration in range(n_batches):
                batch_X = trainX[batch_size*iteration : batch_size*iteration+batch_size, :]
                batch_Y = trainY[batch_size*iteration : batch_size*iteration+batch_size]
                sess.run(train_step, feed_dict = {features: batch_X, labels: batch_Y})
            #if epoch % 10 == 0:
            print "epoch: {}, Training accuracy: {}".format(epoch, sess.run(accuracy, feed_dict={features:trainX,labels:trainY})[1])

        saver.save(sess, "./checkpoint/model.ckpt")
        #saver.restore(sess, "./checkpoint/model.ckpt")

        preds_vals = sess.run(preds, feed_dict={features: testX})
        accuracy = sess.run(accuracy, feed_dict={features: testX, labels: testY})[1]

        precision = precision_score(y_true=testY, y_pred=preds_vals)
        recall = recall_score(y_true=testY, y_pred=preds_vals)

        s,b = compute_s_b(test_weights,testY,preds_vals)
        ams_score = ams(s,b)
        return accuracy, precision, recall, ams_score
        # return preds_vals


def main():
    trainDF = pd.read_csv('higgs-boson/preprocessed_training_data.csv', dtype=float, sep=',', header=None)
    data = trainDF.to_numpy()

    train = data[:, :data.shape[1] - 1]
    labels = data[:, -1].reshape((data.shape[0],1))

    # print train.shape   #(250000,31)
    # print labels.shape  #(250000,1)

    trainX, testX, trainY, testY = train_test_split(train, labels, train_size = 200000, test_size = 50000, random_state=42)

    test_eventIds = testX[:,0].reshape((testX.shape[0],1))
    test_weights = testX[:,-1].reshape((testX.shape[0],1))
    trainX = trainX[:, 1:trainX.shape[1] - 1]
    testX = testX[:, 1:testX.shape[1] - 1]

    max_trainX = np.absolute(np.max(trainX,axis=0).reshape((trainX.shape[1],)))

    trainX = trainX / max_trainX
    testX = testX / max_trainX

    accuracy, precision, recall, ams_score = trainNN(trainX, trainY, testX, testY, test_weights)

    print "Average Test Accuracy: {}".format(accuracy)
    print "Average Precision: {}".format(precision)
    print "Average Recall: {}".format(recall)
    print "Average AMS scores: {}".format(ams_score)



if __name__== "__main__":
    main()
