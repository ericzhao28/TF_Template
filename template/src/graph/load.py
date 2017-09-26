from .graph import Graph


def create_graph(sess, edges, test=False):
  '''
  Create graph based off of text file of tuples linked to
  in path.
  '''

  cls = "main" if test is False else "test"
  graph = Graph(sess)
  for parent, rel, child in edges:
    graph.add_entity(name=parent, cls=cls)
    graph.add_entity(name=child, cls=cls)
    parent_nd = graph.build_node(name=parent, cls=cls, ind="a")
    child_nd = graph.build_node(name=child, cls=cls, ind="b")
    graph.add_relationship(parent_nd, child_nd, rel="friends")
  return graph


def load_graph(sess):
  return Graph(sess)

