from . import Graph


def create_graph(sess, edges):
  '''
  Create graph based off of text file of tuples linked to
  in path.
  '''
  graph = Graph(sess)
  for parent, rel, child in edges:
    graph.add_entity(name=parent, cls="main")
    graph.add_entity(name=child, cls="main")
    parent_nd = graph.build_node(name=parent, cls="main", ind="a")
    child_nd = graph.build_node(name=child, cls="main", ind="b")
    graph.add_relationship(parent_nd, child_nd, rel="friends")
  return graph


def load_graph(sess):
  return Graph(sess)

