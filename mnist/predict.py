import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import model


def test(X_test, Y_test):
    """
    restore the model, and test the model by test dataset

    :param X_test: mnist test dataset
    :param Y_test: mnist test dataset
    :return:
    """

    keep_prob = tf.constant(1.)
    prediction = model.convolution(X_test, keep_prob)
    accuracy = model.compute_accuracy(Y_test, prediction)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('model')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.logging.fatal('No model to restore!!!')
        test_acc = sess.run(accuracy) * 100.
        print("test accuracy: %.2f%%" % test_acc)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    test(mnist.test.images, mnist.test.labels)