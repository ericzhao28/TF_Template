from . import DriverMedium


class Graph(DriverMedium):
  '''
  Graph class for holding basic graph logic.
  '''

  def __init__(self, session):
    self.session = session

  def disconnect(self):
    '''
    Terminate neo4j session.
    '''
    self.session.close()

  @property
  def entity_names(self):
    '''
    Return names of all entities
    '''
    return self.sess.read_transaction(self.tx_get_entities)

  def get_entity(self, name, cls):
    '''
    Get an entity.
    '''
    return self.sess.read_transaction(self.tx_get_entity, name, cls)

  def add_entity(self, name, cls):
    '''
    Add an entity to the graph.
    '''
    self.sess.write_transaction(self.tx_merge_entity, name, cls)

  def add_relationship(self, *args, **kargs):
    pass

  def wipe(self):
    '''
    Wipe all nodes from graph.
    '''
    self.sess.write_transaction(self.tx_delete_entities, None, None)

  def wipe_tests(self):
    '''
    Wipe all test nodes from graph.
    '''
    self.sess.write_transaction(self.tx_delete_entities, None, "test")

