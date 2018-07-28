import tensorflow as tf

from mnist import model


def inference(input):
    """
    Inference the output by input

    :param input: the input data
    :return: the output by model inference
    """

    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, 784])
    prediction = model.convolution(x, keep_prob)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('mnist/model')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            tf.logging.fatal('No model to restore!!!')

        return sess.run(prediction, feed_dict={x:input, keep_prob:1.}).flatten().tolist()


if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('mnist/MNIST_data', one_hot=True)
    prediction = inference(mnist.test.images[0:1])
    print(mnist.test.labels[0:1])
    print(prediction)