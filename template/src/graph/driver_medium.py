class DriverMedium():
  '''
  Offers convenient layer between Neo4j and client
  Node = string.
  Entity = record member.
  Edge = string.
  Relationship = record member.
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
  def build_edge(rel=None, ind="r"):
    '''
    Build edge string to feed into Cypher queries.
    '''
    rel_compound = "" if rel is None else (":%s" % rel)
    return "[" + ind + rel_compound + "]"

  @staticmethod
  def entity_to_node(entity, *args, **kargs):
    '''
    Build node from record entity to feed into Cypher queries.
    '''
    return self.build_node(*args, 
                           cls=entity.labels[0], 
                           properties=entity.properties, 
                           **kargs)

