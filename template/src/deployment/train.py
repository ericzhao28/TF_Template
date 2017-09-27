from ..main import load
from ...datasets import test_set
import tensorflow as tf
from neo4j.v1 import GraphDatabase
from .logger import train_logger


def main():
  '''
  Train primary model.
  '''
  driver = GraphDatabase.driver("bolt://0.0.0.0:7687")
  with driver.session() as neo_sess:
    train_logger.debug('Starting Neo4j session.')
    with tf.Session() as tf_sess:
      train_logger.debug('Starting Tensorflow session.')
      primary_model = load.initialize_models(tf_sess, neo_sess)
      train_logger.debug('Model initialized.')
      dataset = test_set.load()
      train_logger.debug('Datasets loaded.')
      assert(dataset is not None)
      train_logger.debug('Dataset validated.')
      X, Y = dataset
      primary_model.train(X, Y)
      train_logger.debug('Primary model trained.')
      primary_model.save()
      train_logger.debug('Model saved.')


if __name__ == "__main__":
  train_logger.info('Starting primary model training.')
  main()
  train_logger.info('Primary model training complete :)')
