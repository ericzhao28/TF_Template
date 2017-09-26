from .graph import Graph


def create_graph(sess, edges, test=False):
  '''
  Create graph based off of text file of tuples linked to
  in path.
  '''

  cls = "main" if test is False else "test"
  graph = Graph(sess)
  for parent, rel, child in edges:
    parent_nd = graph.build_node(name=parent, cls=cls, ind='a')
    child_nd = graph.build_node(name=child, cls=cls, ind='b')
    edge = graph.build_edge(rel=rel)
    graph.add_entity(parent_nd)
    graph.add_entity(child_nd)
    graph.add_relationship(parent_nd, child_nd, edge)
  return graph


def load_graph(sess):
  '''
  Generate empty graph without loading in additional data,
  assuming server-side Neo4j graph already exists.
  '''
  return Graph(sess)

