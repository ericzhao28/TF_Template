from ..main import load
from ...datasets import test_set
import tensorflow as tf
from neo4j.v1 import GraphDatabase


def main():
  '''
  Train primary model.
  '''
  driver = GraphDatabase.driver("bolt://0.0.0.0:7687")
  with driver.session() as neo_sess:
    with tf.Session() as tf_sess:
      primary_model = load.initialize_models(tf_sess, neo_sess)
      dataset = test_set.load()
      assert(dataset is not None)
      X, Y = dataset
      primary_model.train(X, Y)
      primary_model.save()


if __name__ == "__main__":
  main()
