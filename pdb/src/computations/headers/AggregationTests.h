#pragma once

#include <KeyExtractionLambda.h>

namespace pdb {

// SFINAE test
template <typename T>
class hasKeyProjection
{
  typedef char one;
  struct two { char x[2]; };

  template <typename C> static one test( typeof(&C::getKeyProjectionWithInputKey) ) ;
  template <typename C> static two test(...);

 public:
  enum { value = sizeof(test<T>(0)) == sizeof(char) };
};

template <typename Derived, typename InputClass, typename KeyClass>
typename std::enable_if<hasKeyProjection<Derived>::value, Lambda<KeyClass>>::type
callGetKeyProjection (Derived* callOnMe) {

  // make sure everything is fine
  using key = typename std::remove_reference<decltype(((InputClass*) nullptr)->getKey())>::type;
  static_assert(std::is_base_of<HandleBase, key>::value, "The input key has to be a Handle!");

  // get the predicate
  GenericHandle first (1);
  auto predicate = callOnMe->getKeyProjectionWithInputKey (first);

  // prepare the input
  Handle<InputClass> in = first;

  // inject the key lambda
  predicate.inject(0, LambdaTree<Ptr<InputClass>>(std::make_shared<KeyExtractionLambda<InputClass>>(in, 0)));

  return predicate;
}

template <typename Derived, typename InputClass, typename KeyClass>
typename std::enable_if<!hasKeyProjection<Derived>::value, Lambda<KeyClass>>::type
callGetKeyProjection (Derived* callOnMe) {

  GenericHandle first (1);
  return callOnMe->getKeyProjection (first);
}

template <typename Derived, typename InputClass, typename KeyClass>
typename std::enable_if<hasKeyProjection<Derived>::value, Lambda<KeyClass>>::type
callGetKeyProjectionWithKey (Derived* callOnMe) {

  // get the predicate
  GenericHandle first (1);
  return callOnMe->getKeyProjectionWithInputKey (first);
}

template <typename Derived, typename InputClass, typename KeyClass>
typename std::enable_if<!hasKeyProjection<Derived>::value, Lambda<KeyClass>>::type
callGetKeyProjectionWithKey (Derived* callOnMe) {
  throw runtime_error("This aggregation does not support a key only TCAP");
}
}
