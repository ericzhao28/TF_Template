from ..models import SeqAttModel
from ...credentials import azure_account_name, azure_account_key
from azure.storage.blob import BlockBlobService
import tensorflow as tf


def initialize_func(sess, logger, FLAGS):
  '''
  Initialize main function to predict using model.
  '''

  tf.set_random_seed(1)

  ##############################
  ### Log hyperparameters.
  param_desc = FLAGS.model_name + ":   "
  for flag, val in FLAGS.__dict__['__flags'].items():
    param_desc += flag + ": " + str(val) + "; "
  logger.debug("Parameters " + param_desc)
  ##############################

  ##############################
  ### Instantiate model.
  model = SeqAttModel(sess, FLAGS, logger,
                      model_name=FLAGS.model_name)
  logger.debug('Model instantiated.')
  ##############################

  ##############################
  ### Download model
  block_blob_service = BlockBlobService(
      account_name=azure_account_name,
      account_key=azure_account_key
  )

  for suffix in [".meta", ".index", ".data-00000-of-00001"]:
    filename = FLAGS.model_name + "-" + str(FLAGS.iter_num) + suffix
    logger.debug("Loading save file: " + filename)
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
  logger.debug('Model initialized.')
  model.restore(FLAGS.checkpoints_dir + FLAGS.model_name +
                "-" + FLAGS.iter_num)
  logger.debug('Model restored.')
  ##############################

  def function(query):
    result = model.predict([query])[0]
    logger.debug('Model predicted: ' + str(result))
    return result

  return function


