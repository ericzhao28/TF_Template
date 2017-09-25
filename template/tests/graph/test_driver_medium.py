from ...src.graph import DriverMedium


def test_build_node():
  driver = DriverMedium()
  assert("(n)" == driver.build_node())
  assert("(a)" == driver.build_node(ind="a"))
  assert("(a:test)" == driver.build_node(cls="test", ind="a"))
  assert("(n:test)" == driver.build_node(cls="test", ind="n"))
  assert("(n {name:'yay'})" == driver.build_node(name="yay"))
  assert("(n {name:'yay', woo:'hello', no:'what'})" ==
         driver.build_node(name="yay", properties={'woo': 'hello',
                                                   'no': 'what'}))
  assert("(n {woo:'hello', no:'what'})" ==
         driver.build_node(properties={'woo': 'hello', 'no': 'what'}))
  assert("(n:test {name:'yay', woo:'hello', no:'what'})" ==
         driver.build_node(cls="test", name="yay", properties={'woo': 'hello',
                                                               'no': 'what'}))
  assert("(n:test {woo:'hello', no:'what'})" ==
         driver.build_node(cls="test", properties={'woo': 'hello',
                                                   'no': 'what'}))
  assert("(a:test {woo:'hello', no:'what'})" ==
         driver.build_node(cls="test", properties={'woo': 'hello',
                                                   'no': 'what'}, ind="a"))


def test_build_relationship():
  driver = DriverMedium()
  assert("[r]" == driver.build_relationship())
  assert("[r:friends]" == driver.build_relationship("friends"))
  assert("[b]" == driver.build_relationship(ind="b"))
  assert("[b:friends]" == driver.build_relationship("friends", "b"))

