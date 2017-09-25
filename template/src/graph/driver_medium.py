class DriverMedium():
  '''
  Offers convenient layer between Neo4j and client
  '''

  @staticmethod
  def tx_get_entity(tx, name=None, cls=None):
    '''
    Get entity with optional name and class.
    '''
    if name is None:
      name_compound = ""
    else:
      name_compound = " {name: '%s'}" % name

    if cls is None:
      cls_compound = ""
    else:
      cls_compound = ":%s" % cls

    result = tx.run("MATCH (n$cls$name) RETURN n", cls=cls_compound,
                    name=name_compound)
    return result.single()[0]

  @staticmethod
  def tx_get_entities(tx, name=None, cls=None):
    '''
    Get entity with optional name and class.
    '''
    if name is None:
      name_compound = ""
    else:
      name_compound = " {name: '%s'}" % name

    if cls is None:
      cls_compound = ""
    else:
      cls_compound = ":%s" % cls

    result = tx.run("MATCH (n$cls$name) RETURN n", cls=cls_compound,
                    name=name_compound)
    return result.all()

  @staticmethod
  def tx_create_entity(tx, name=None, cls=None):
    '''
    Create entity with optional name and class.
    '''
    if name is None:
      name_compound = ""
    else:
      name_compound = " {name: '%s'}" % name

    if cls is None:
      cls_compound = ""
    else:
      cls_compound = ":%s" % cls

    tx.run("CREATE (n$cls$name)", cls=cls_compound, name=name_compound)
    return None

  @staticmethod
  def tx_merge_entity(tx, name=None, cls=None):
    '''
    Merge entity with optional name and class.
    '''
    if name is None:
      name_compound = ""
    else:
      name_compound = " {name: '%s'}" % name

    if cls is None:
      cls_compound = ""
    else:
      cls_compound = ":%s" % cls

    tx.run("MERGE (n$cls$name)", cls=cls_compound, name=name_compound)
    return None

  @staticmethod
  def tx_delete_entities(tx, name=None, cls=None):
    '''
    Delete entities with optional name and class.
    '''
    if name is None:
      name_compound = ""
    else:
      name_compound = " {name: '%s'}" % name

    if cls is None:
      cls_compound = ""
    else:
      cls_compound = ":%s" % cls

    tx.run("MATCH (n$cls$name) DETACH DELETE n", cls=cls_compound,
           name=name_compound)
    return None

