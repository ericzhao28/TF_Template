class DriverMedium():
  '''
  Offers convenient layer between Neo4j and client
  '''

  @staticmethod
  def build_node(name=None, cls=None, properties={}, ind="n"):
    '''
    Build node string to feed into Cypher queries.
    '''
    cls_compound = "" if cls is None else (":%s" % cls)
    if name is not None:
      properties['name'] = name

    if len(properties) > 0:
      property_compound = " {"
      for key, value in properties.items():
        property_compound += "%s:'%s', " % (key, value)
      property_compound = property_compound[:-2] + "}"
    else:
      property_compound = ""

    return "(" + ind + cls_compound + property_compound + ")"

  @staticmethod
  def build_relationship(rel=None, ind="r"):
    '''
    Build relationship string to feed into Cypher queries.
    '''
    rel_compound = "" if rel is None else (":%s" % rel)
    return "[" + ind + rel_compound + "]"

  @staticmethod
  def tx_get_relationship(tx, parent, child, relationship):
    '''
    Get entity.
    '''
    result = tx.run("MATCH %s-%s->%s RETURN n" % (child, relationship, parent))
    return result.single()[0]

  @staticmethod
  def tx_get_entity(tx, node):
    '''
    Get entity.
    '''
    result = tx.run("MATCH %s RETURN n" % node)
    return result.single()[0]

  @staticmethod
  def tx_get_entities(tx, node):
    '''
    Get entities.
    '''

    result = tx.run("MATCH %s RETURN n" % node)
    return result.single()

  @staticmethod
  def tx_create_relationship(tx, parent, child, relationship):
    '''
    Get entity.
    '''
    tx.run("CREATE %s-%s->%s" % (child, relationship, parent))
    return None

  @staticmethod
  def tx_create_entity(tx, node):
    '''
    Create entity.
    '''
    tx.run("CREATE %s" % node)
    return None

  @staticmethod
  def tx_merge_relationship(tx, parent, child, relationship):
    '''
    Get entity.
    '''
    tx.run("MERGE %s-%s->%s" % (child, relationship, parent))
    return None

  @staticmethod
  def tx_merge_entity(tx, node):
    '''
    Merge entity.
    '''
    tx.run("MERGE %s" % node)
    return None

  @staticmethod
  def tx_delete_entities(tx, node):
    '''
    Delete entities with optional name and class.
    '''
    tx.run("MATCH %s DETACH DELETE n" % node)
    return None

  @staticmethod
  def tx_delete_relationships(tx, parent, child, relationship):
    '''
    Delete entities with optional name and class.
    '''
    tx.run("MATCH %s-%s->%s DETACH DELETE r" % (child, relationship, parent))
    return None

