from ...src.models import SeqAttModel
from ...datasets import example
from ...credentials import azure_account_name, azure_account_key
from .logger import eval_logger
from . import config
from azure.storage.blob import BlockBlobService
import tensorflow as tf


def evaluate(FLAGS):
  with tf.Session() as sess:
    tf.set_random_seed(1)

    ##############################
    ### Log hyperparameters.
    param_desc = FLAGS.model_name + ":   "
    for flag, val in FLAGS.__dict__['__flags'].items():
      param_desc += flag + ": " + str(val) + "; "
    eval_logger.debug("Parameters " + param_desc)
    ##############################

    ##############################
    ### Instantiate model.
    model = SeqAttModel(sess, FLAGS, eval_logger,
                        model_name=FLAGS.model_name)
    eval_logger.debug('Model instantiated.')
    ##############################

    ##############################
    ### Load dataset.
    test_dataset = example.load(None)
    eval_logger.debug('Dataset loaded.')
    ##############################

    ##############################
    ### Download model
    block_blob_service = BlockBlobService(
        account_name=azure_account_name,
        account_key=azure_account_key
    )

    for suffix in [".meta", ".index", ".data-00000-of-00001"]:
      filename = FLAGS.model_name + "-" + str(FLAGS.iter_num) + suffix
      eval_logger.debug("Loading save file: " + filename)
      assert(filename in [x.name for x in
                          block_blob_service.list_blobs("models")])
      block_blob_service.get_blob_to_path(
          "models",
          filename,
          FLAGS.checkpoints_dir + filename
      )
    ##############################

    ##############################
    ### Build model
    model.initialize()
    eval_logger.debug('Model initialized.')
    model.restore(FLAGS.checkpoints_dir + FLAGS.model_name +
                  "-" + FLAGS.iter_num)
    eval_logger.debug('Model restored.')
    ##############################

    ##############################
    ### Evaluate
    loss, acc, tpr, fpr, summary = model.evaluate(
        test_dataset[0], test_dataset[1], prefix="test")
    eval_logger.info(
        FLAGS.model_name +
        "; loss: %f, accuracy: %f, TPR: %s, FPR: %s"
        % (loss, acc, str(tpr), str(fpr)))
    print(FLAGS.model_name + ": testing complete.")
    ##############################


if __name__ == "__main__":
  FLAGS = tf.app.flags.FLAGS

  tf.app.flags.DEFINE_string("model_name", "default",
                             "Name of model to be used in logs.")
  tf.app.flags.DEFINE_string("iter_num", "0",
                             "Iteration number of save.")
  tf.app.flags.DEFINE_integer("s_batch", 32,
                              "Size of batches")
  tf.app.flags.DEFINE_float("v_regularization", 0.15,
                            "Value of regularization term")

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

  evaluate(FLAGS)

