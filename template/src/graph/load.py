from . import Graph


def create_graph(sess, edges):
  '''
  Create graph based off of text file of tuples linked to
  in path.
  '''
  graph = Graph(sess)
  for parent, rel, child in edges:
    graph.add_entity(parent, "main")
    parent_nd = graph.get_entity(parent, "main")
    graph.add_entity(child, "main")
    child_nd = graph.get_entity(child, "main")
    graph.add_relationship(parent_nd, child_nd, rel)
  return graph


def load_graph(sess):
  return Graph(sess)

