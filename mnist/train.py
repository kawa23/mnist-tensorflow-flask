import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import model


def main():
    """

    :return:
    """

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    minibatch_size = 128
    minibatch_num = mnist.train.num_examples // minibatch_size
    # learning rate decay
    train_step = 0
    init_lr = 1e-3
    global_ = tf.Variable(tf.constant(0))

    keep_prob = tf.placeholder(tf.float32)
    X = tf.placeholder(tf.float32, [None, 784])
    Y = tf.placeholder(tf.float32, [None, 10])
    prediction = model.convolution(X, keep_prob)

    cross_entropy = model.compute_cost(Y, prediction)
    learning_rate = tf.train.exponential_decay(init_lr, global_, 10, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    accuracy = model.compute_accuracy(Y, prediction)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('model')
        initial_epoch = 1
        if ckpt and ckpt.model_checkpoint_path:
            # recover the model from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            initial_epoch = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        print(initial_epoch)
        for epoch in range(initial_epoch, 100):
            # train the model
            print('training epoch: {}'.format(epoch))
            for minibatch in range(minibatch_num):
                minibatch_xs, minibatch_ys = mnist.train.next_batch(minibatch_size)
                train_step += 1
                _, train_acc, lr = sess.run([optimizer, accuracy, learning_rate],
                                            feed_dict={X: minibatch_xs, Y: minibatch_ys, keep_prob: 0.5,
                                                       global_: train_step})

                if minibatch % 100 == 0:
                    # display the train accuracy
                    print("iter %3d:\tlearning rate=%f,\ttraining accuracy=%.2f%%" % (minibatch, lr, train_acc * 100))

            # run validation after every epoch
            validation_acc = sess.run(accuracy, feed_dict={X: mnist.validation.images, Y: mnist.validation.labels, keep_prob: 1.})
            print('---------------------------------------------------------')
            print("epoch: %3d, validation accuracy: %.2f%%" % (epoch, validation_acc * 100))
            print('---------------------------------------------------------')
            # save the model
            saver.save(sess, './model/my-model', global_step=epoch)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.ERROR)
    main()