from .driver_medium import DriverMedium


class Graph(DriverMedium):
  '''
  Graph class for holding basic graph logic.
  '''

  def __init__(self, sess):
    self.sess = sess

  def disconnect(self):
    '''
    Terminate neo4j session.
    '''
    self.sess.close()

  @property
  def entity_names(self):
    '''
    Return names of all entities
    '''
    return self.sess.read_transaction(self.tx_get_entities, self.build_node())

  def get_entity(self, *args, **kargs):
    '''
    Get an entity.
    '''
    return self.sess.read_transaction(self.tx_get_entity,
                                      self.build_node(*args, **kargs))

  def add_entity(self, *args, **kargs):
    '''
    Add an entity to the graph.
    '''
    self.sess.write_transaction(self.tx_merge_entity,
                                self.build_node(*args, **kargs))

  def add_relationship(self, parent, child, *args, **kargs):
    '''
    Add a relationship to the graph.
    '''
    self.sess.write_transaction(self.tx_merge_relationship,
                                parent, child,
                                self.build_relationship(*args, **kargs))

  def wipe(self):
    '''
    Wipe all nodes from graph.
    '''
    self.sess.write_transaction(self.tx_delete_entities, self.build_node())

  def wipe_tests(self):
    '''
    Wipe all test nodes from graph.
    '''
    self.sess.write_transaction(self.tx_delete_entities,
                                self.build_node(cls="test"))

