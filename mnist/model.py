import tensorflow as tf



def convolution(x, keep_prob):
    """
    Multilayer Convolutional Network

    :param x: dataset of input
    :param keep_prob: keep probability for the dropout
    :return:
    """

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def weight_variable(name, shape):
        return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(seed=0))

    def bias_variable(name, shape):
        return tf.get_variable(name, shape, initializer=tf.zeros_initializer())

    # first convolution layer
    x_image = tf.reshape(x, [-1,28,28,1],)
    W_conv1 = weight_variable('W_conv1', [5,5,1,32])
    b_conv1 = bias_variable('b_conv1', [32])
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1))
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolution layer
    W_conv2 = weight_variable('W_conv2', [5,5,32,64])
    b_conv2 = bias_variable('b_conv2', [64])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, W_conv2), b_conv2))
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable('W_fc1', [7*7*64, 1024])
    b_fc1 = bias_variable('b_fc1', [1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # output layer
    W_fc2 = weight_variable('W_fc2', [1024, 10])
    b_fc2 = bias_variable('b_fc2', [10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

    return y


def compute_cost(Y, prediction):
    """
    compute the cost for the model

    :param Y: the labels of the dataset
    :param prediction: the prediction of the model output
    :return: the cost of the model
    """

    cost = -tf.reduce_sum(Y * tf.log(prediction))

    return cost


def compute_accuracy(Y, prediction):
    """
    compute the accuracy for the model

    :param Y: the labels of the dataset
    :param prediction: the prediction of the model output
    :return: the accuracy of the model
    """
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(prediction, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy