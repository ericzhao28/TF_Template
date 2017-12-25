from flask import request, jsonify
from flask_api import FlaskAPI
from flask_cors import CORS
from ..main import load
from .logger import api_logger
from . import config
import tensorflow as tf

app = FlaskAPI(__name__)
cors = CORS(app)


with tf.Session() as sess:
  execute_func = None

  @app.route("/demo", methods=['POST'])
  def get_demo():
    """
    Simple synchronous endpoint for demo.

    Make query to 0.0.0.0:5000/query with
    POST protocol, payload: {"query": "hi"}.
    Response: {"response": "hello"}.
    """

    text = str(request.data.get('query', ''))
    api_logger.debug('Received query: ' + text)
    response = execute_func(text)
    api_logger.debug('Responding with: ' + response)
    return jsonify({'response': response})

  if __name__ == "__main__":
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string("model_name", "default",
                               "Name of model to be used in logs.")
    tf.app.flags.DEFINE_string("iter_num", "0",
                               "Iteration number of save.")
    tf.app.flags.DEFINE_integer("s_batch", 32,
                                "Size of batches")

    tf.app.flags.DEFINE_integer("n_features", 77,
                                "Number of features")
    tf.app.flags.DEFINE_integer("n_steps", 22,
                                "Number of steps in input sequence")

    tf.app.flags.DEFINE_integer("h_gru", 64,
                                "Hidden units in GRU layer")
    tf.app.flags.DEFINE_integer("h_att", 16,
                                "Hidden units in attention mechanism")
    tf.app.flags.DEFINE_integer("o_gru", 64,
                                "Output units in GRU layer")
    tf.app.flags.DEFINE_integer("h_dense", 64,
                                "Hidden units in first dense layer")
    tf.app.flags.DEFINE_integer("o_dense", 32,
                                "Output units in first dense layer")
    tf.app.flags.DEFINE_integer("h_dense2", 32,
                                "Hidden units in second dense layer")
    tf.app.flags.DEFINE_integer("o_dense2", 16,
                                "Output units in second dense layer")
    tf.app.flags.DEFINE_integer("n_classes", 2,
                                "Number of label classes")

    tf.app.flags.DEFINE_string("graphs_train_dir", config.GRAPHS_TRAIN_DIR,
                               "Graph train directory")
    tf.app.flags.DEFINE_string("graphs_test_dir", config.GRAPHS_TEST_DIR,
                               "Graph test directory")
    tf.app.flags.DEFINE_string("checkpoints_dir", config.CHECKPOINTS_DIR,
                               "Checkpoints directory")

    execute_func = load.initialize_func(sess, api_logger, FLAGS)
    api_logger.info('Starting deployment API on 0.0.0.0...')
    app.run(host='0.0.0.0')

