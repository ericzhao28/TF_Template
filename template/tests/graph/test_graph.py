from ...src.graph import Graph, load
from neo4j.v1 import GraphDatabase


def test_graph():
  driver = GraphDatabase.driver("bolt://0.0.0.0:7687")
  with driver.session() as neo_sess:
    graph = Graph(neo_sess)
    graph.wipe_tests()
    edges = [("parent1", "dad", "child1"),
             ("parent2", "mom", "child1"),
             ("parent1", "dad", "child2"),
             ("parent2", "mom", "child2"),
             ("parent1", "uncle", "child3"),
             ("parent2", "aunt", "child3")]

    for parent, rel, child in edges:
      graph.add_entity(name=parent, cls="test")
      graph.add_entity(name=child, cls="test")
      parent_nd = graph.build_node(name=parent, cls="test", ind="a")
      child_nd = graph.build_node(name=child, cls="test", ind="b")
      graph.add_relationship(parent_nd, child_nd, rel="friends")

    assert(set(graph.entity_names) == set(["child1", "child2",
                                          "child3", "parent1",
                                          "parent2"]))
    assert(graph.get_entity(name="child1"))
    assert(not graph.get_entity(name="sdf"))
    graph.wipe_tests()
    assert(not graph.get_entity(name="child1"))
    assert(graph.entity_names == [])


def test_load():
  driver = GraphDatabase.driver("bolt://0.0.0.0:7687")
  with driver.session() as neo_sess:
    edges = [("parent1", "dad", "child1"),
             ("parent2", "mom", "child1"),
             ("parent1", "dad", "child2"),
             ("parent2", "mom", "child2"),
             ("parent1", "uncle", "child3"),
             ("parent2", "aunt", "child3")]
    graph = load.create_graph(neo_sess, edges, test=True)
    graph.wipe_tests()

    assert(set(graph.entity_names) == set(["child1", "child2",
                                          "child3", "parent1",
                                          "parent2"]))
    assert(graph.get_entity(name="child1"))
    assert(not graph.get_entity(name="sdf"))

  driver = GraphDatabase.driver("bolt://0.0.0.0:7687")
  with driver.session() as neo_sess:
    graph = load.create_graph(neo_sess)

    assert(set(graph.entity_names) == set("child1", "child2",
                                          "child3", "parent1",
                                          "parent2"))
    assert(graph.get_entity(name="child1"))
    assert(not graph.get_entity(name="sdf"))
    graph.wipe_tests()
    assert(not graph.get_entity(name="child1"))
    assert(graph.entity_names == [])

