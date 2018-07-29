import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import numpy as np

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


app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/inference', methods=['POST'])
def mnist():
    # MNIST data white and black value are opposite
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    prediction = inference(input)
    # print(prediction)
    return jsonify(results=prediction)


if __name__ == '__main__':
    app.run()