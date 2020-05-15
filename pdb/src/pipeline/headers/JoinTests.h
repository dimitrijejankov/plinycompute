/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#pragma once

#include <boost/tti/has_member_function.hpp>
#include <lambdas/JoinRecordLambda.h>
#include "Handle.h"
#include "ValueExtractionLambda.h"

// all of this nastiness allows us to call getSelection and getProjection on a join, using the correct number of args
namespace pdb {

extern GenericHandle foofoo;

struct HasTwoArgs {

  template <typename U>
  static auto testSelection (U *x) -> decltype (x->getSelection (foofoo, foofoo)) {
    return x->getSelection (foofoo, foofoo);
  }

  template <typename U>
  static auto testKeySelection (U *x) -> decltype (x->getKeySelection (foofoo, foofoo)) {
    return x->getKeySelection (foofoo, foofoo);
  }

  template <typename U>
  static auto testProjection (U *x) -> decltype (x->getProjection (foofoo, foofoo)) {
    return x->getProjection (foofoo, foofoo);
  }

  template <typename U>
  static auto testKeyProjection (U *x) -> decltype (x->getKeyProjection (foofoo, foofoo)) {
    return x->getKeyProjection (foofoo, foofoo);
  }

  template <typename U>
  static auto testValueProjection (U *x) -> decltype (x->getValueProjection (foofoo, foofoo)) {
    return x->getValueProjection (foofoo, foofoo);
  }
};

struct HasThreeArgs {

  template <typename U>
  static auto testSelection (U *x) -> decltype (x->getSelection (foofoo, foofoo, foofoo)) {
    return x->getSelection (foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testKeySelection (U *x) -> decltype (x->getKeySelection (foofoo, foofoo, foofoo)) {
    return x->getKeySelection (foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testProjection (U *x) -> decltype (x->getProjection (foofoo, foofoo, foofoo)) {
    return x->getProjection (foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testKeyProjection (U *x) -> decltype (x->getKeyProjection (foofoo, foofoo, foofoo)) {
    return x->getKeyProjection (foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testValueProjection (U *x) -> decltype (x->getValueProjection (foofoo, foofoo, foofoo)) {
    return x->getValueProjection (foofoo, foofoo, foofoo);
  }
};

struct HasFourArgs {

  template <typename U>
  static auto testSelection (U *x) -> decltype (x->getSelection (foofoo, foofoo, foofoo, foofoo)) {
    return x->getSelection (foofoo, foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testKeySelection (U *x) -> decltype (x->getKeySelection (foofoo, foofoo, foofoo, foofoo)) {
    return x->getKeySelection (foofoo, foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testProjection (U *x) -> decltype (x->getProjection (foofoo, foofoo, foofoo, foofoo)) {
    return x->getProjection (foofoo, foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testKeyProjection (U *x) -> decltype (x->getKeyProjection (foofoo, foofoo, foofoo, foofoo)) {
    return x->getKeyProjection (foofoo, foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testValueProjection (U *x) -> decltype (x->getValueProjection (foofoo, foofoo, foofoo, foofoo)) {
    return x->getValueProjection (foofoo, foofoo, foofoo, foofoo);
  }
};

struct HasFiveArgs {

  template <typename U>
  static auto testSelection (U *x) -> decltype (x->getSelection (foofoo, foofoo, foofoo, foofoo, foofoo)) {
    return x->getSelection (foofoo, foofoo, foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testKeySelection (U *x) -> decltype (x->getKeySelection (foofoo, foofoo, foofoo, foofoo, foofoo)) {
    return x->getKeySelection (foofoo, foofoo, foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testProjection (U *x) -> decltype (x->getProjection (foofoo, foofoo, foofoo, foofoo, foofoo)) {
    return x->getProjection (foofoo, foofoo, foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testKeyProjection (U *x) -> decltype (x->getKeyProjection (foofoo, foofoo, foofoo, foofoo, foofoo)) {
    return x->getKeyProjection (foofoo, foofoo, foofoo, foofoo, foofoo);
  }

  template <typename U>
  static auto testValueProjection (U *x) -> decltype (x->getValueProjection (foofoo, foofoo, foofoo, foofoo, foofoo)) {
    return x->getValueProjection (foofoo, foofoo, foofoo, foofoo, foofoo);
  }
};

/**
 * This test checks if the key selection even exists
 */

struct KeySelectionExists {

  template <typename U>
  static true_type check(U* arg, decltype (HasTwoArgs::testKeySelection (arg)) *arg2 = nullptr) {
    return true_type {};
  }

  template <typename U>
  static true_type check(U* arg, decltype (HasThreeArgs::testKeySelection (arg)) *arg2 = nullptr) {
    return true_type {};
  }

  template <typename U>
  static true_type check(U* arg, decltype (HasFourArgs::testKeySelection (arg)) *arg2 = nullptr) {
    return true_type {};
  }

  template <typename U>
  static true_type check(U* arg, decltype (HasFiveArgs::testKeySelection (arg)) *arg2 = nullptr) {
    return true_type {};
  }

  template <typename U>
  static false_type check(U* arg) {
    return false_type {};
  }

};


/**
 *
 */

struct KeyProjectionExists {

  template <typename U>
  static true_type check(U* arg, decltype (HasTwoArgs::testKeyProjection (arg)) *arg2 = nullptr) {
    return true_type {};
  }

  template <typename U>
  static true_type check(U* arg, decltype (HasThreeArgs::testKeyProjection (arg)) *arg2 = nullptr) {
    return true_type {};
  }

  template <typename U>
  static true_type check(U* arg, decltype (HasFourArgs::testKeyProjection (arg)) *arg2 = nullptr) {
    return true_type {};
  }

  template <typename U>
  static true_type check(U* arg, decltype (HasFiveArgs::testKeyProjection (arg)) *arg2 = nullptr) {
    return true_type {};
  }

  template <typename U>
  static false_type check(U* arg) {
    return false_type {};
  }

};

/**
 *
 */

template <typename LambdaType, typename In1, typename ...Rest>
typename std::enable_if<sizeof ...(Rest) != 0, void>::type
injectKeyExtraction(LambdaType predicate, int input) {

  injectKeyExtraction<LambdaType, Rest...>(predicate, input + 1);

  // prepare the input
  GenericHandle tmp(input + 1);
  Handle<In1> in = tmp;

  // inject the key lambda
  predicate.inject(input, LambdaTree<Ptr<In1>>(std::make_shared<KeyExtractionLambda<In1>>(in, input)));
}

template <typename LambdaType, typename In1>
void injectKeyExtraction(LambdaType predicate, int input) {

  // prepare the input
  GenericHandle tmp(input + 1);
  Handle<In1> in = tmp;

  // inject the key lambda
  predicate.inject(input, LambdaTree<Ptr<In1>>(std::make_shared<KeyExtractionLambda<In1>>(in, input)));
}

/**
 *
 */

template <typename LambdaType, typename In1, typename ...Rest>
typename std::enable_if<sizeof ...(Rest) != 0, void>::type
injectValueExtraction(LambdaType predicate, int input) {

  injectValueExtraction<LambdaType, Rest...>(predicate, input + 1);

  // prepare the input
  GenericHandle tmp(input + 1);
  Handle<In1> in = tmp;

  // inject the key lambda
  predicate.inject(input, LambdaTree<Ptr<In1>>(std::make_shared<ValueExtractionLambda<In1>>(in)));
}

template <typename LambdaType, typename In1>
void injectValueExtraction(LambdaType predicate, int input) {

  // prepare the input
  GenericHandle tmp(input + 1);
  Handle<In1> in = tmp;

  // inject the key lambda
  predicate.inject(input, LambdaTree<Ptr<In1>>(std::make_shared<ValueExtractionLambda<In1>>(in)));
}

/**
 *
 */

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetSelection (TypeToCallMethodOn &a, decltype (HasTwoArgs::testSelection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  return a.getSelection (first, second);
}

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetSelection (TypeToCallMethodOn &a, decltype (HasThreeArgs::testSelection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  return a.getSelection (first, second, third);
}

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetSelection (TypeToCallMethodOn &a, decltype (HasFourArgs::testSelection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  GenericHandle fourth (4);
  return a.getSelection (first, second, third, fourth);
}

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetSelection (TypeToCallMethodOn &a, decltype (HasFiveArgs::testSelection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  GenericHandle fourth (4);
  GenericHandle fifth (5);
  return a.getSelection (first, second, third, fourth, fifth);
}

/**
 *
 */

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetSelection (TypeToCallMethodOn &a, decltype (HasTwoArgs::testKeySelection (&a)) *arg = nullptr) {

  // the arguments
  GenericHandle first (1);
  GenericHandle second (2);

  // call the selection
  auto predicate = a.getKeySelection (first, second);

  // inject the key extraction into the predicate
  injectKeyExtraction<decltype(predicate), In1, In2, Rest...> (predicate, 0);

  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In1*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In2*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");

  // return the predicate
  return predicate;
}

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetSelection (TypeToCallMethodOn &a, decltype (HasThreeArgs::testKeySelection (&a)) *arg = nullptr) {

  // the arguments
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);

  // call the selection
  auto predicate = a.getKeySelection (first, second, third);

  // inject the key extraction into the predicate
  injectKeyExtraction<decltype(predicate), In1, In2, Rest...> (predicate, 0);

  // get the third input
  using In3 = typename std::tuple_element<0, std::tuple<Rest...>>::type;

  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In1*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In2*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In3*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");

  // return the predicate
  return predicate;
}

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetSelection (TypeToCallMethodOn &a, decltype (HasFourArgs::testKeySelection (&a)) *arg = nullptr) {

  // the arguments
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  GenericHandle fourth (4);

  // call the selection
  auto predicate = a.getKeySelection (first, second, third, fourth);

  // inject the key extraction into the predicate
  injectKeyExtraction<decltype(predicate), In1, In2, Rest...> (predicate, 0);

  // get the third input
  using In3 = typename std::tuple_element<0, std::tuple<Rest...>>::type;
  using In4 = typename std::tuple_element<0, std::tuple<Rest...>>::type;

  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In1*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In2*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In3*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In4*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");

  // return the predicate
  return predicate;
}

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetSelection (TypeToCallMethodOn &a, decltype (HasFiveArgs::testKeySelection (&a)) *arg = nullptr) {

  // the arguments
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  GenericHandle fourth (4);
  GenericHandle fifth (5);

  // call the selection
  auto predicate = a.getKeySelection (first, second, third, fourth, fifth);

  // inject the key extraction into the predicate
  injectKeyExtraction<decltype(predicate), In1, In2, Rest...> (predicate, 0);

  // get the third input
  using In3 = typename std::tuple_element<0, std::tuple<Rest...>>::type;
  using In4 = typename std::tuple_element<0, std::tuple<Rest...>>::type;
  using In5 = typename std::tuple_element<0, std::tuple<Rest...>>::type;

  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In1*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In2*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In3*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In4*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, typename std::remove_reference<decltype(((In5*) nullptr)->getKey())>::type>::value, "The key of the input has to be a Handle!");

  // return the predicate
  return predicate;
}

/**
 *
 */

template <typename TypeToCallMethodOn, typename Out, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasTwoArgs::testProjection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  return a.getProjection (first, second);
}

template <typename TypeToCallMethodOn, typename Out, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasThreeArgs::testProjection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  return a.getProjection (first, second, third);
}

template <typename TypeToCallMethodOn, typename Out, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasFourArgs::testProjection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  GenericHandle fourth (4);
  return a.getProjection (first, second, third, fourth);
}

template <typename TypeToCallMethodOn, typename Out, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasFiveArgs::testProjection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  GenericHandle fourth (4);
  GenericHandle fifth (5);
  return a.getProjection (first, second, third, fourth, fifth);
}

/**
 *
 */

template <typename TypeToCallMethodOn, typename Out, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasTwoArgs::testKeyProjection (&a)) *arg1 = nullptr, decltype (HasTwoArgs::testValueProjection (&a)) *arg2 = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);

  // get the key and value projections
  auto valueProjection = a.getValueProjection (first, second);
  auto keyProjection = a.getKeyProjection (first, second);

  // inject the key extraction into the predicate
  injectKeyExtraction<decltype(keyProjection), In1, In2, Rest...> (keyProjection, 0);

  // inject the key extraction into the predicate
  injectValueExtraction<decltype(valueProjection), In1, In2, Rest...> (valueProjection, 0);

  // the types for key and value
  using key = typename std::remove_reference<decltype(((Out*) nullptr)->getKey())>::type;
  using value = typename std::remove_reference<decltype(((Out*) nullptr)->getValue())>::type;

  // make sure the the type is fine
  static_assert(std::is_base_of<HandleBase, key>::value, "The key has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, value>::value, "The value has to be a Handle!");

  // make the join record lambda
  auto lambda = std::make_shared<JoinRecordLambda<Out, key, value>>(keyProjection.tree, valueProjection.tree);

  // create
  return LambdaTree<Handle<Out>>(lambda);
}

template <typename TypeToCallMethodOn, typename Out, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasThreeArgs::testKeyProjection (&a)) *arg1 = nullptr, decltype (HasThreeArgs::testValueProjection (&a)) *arg2 = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);

  // get the key and value projections
  auto valueProjection = a.getValueProjection (first, second, third);
  auto keyProjection = a.getKeyProjection (first, second, third);

  // inject the key extraction into the predicate
  injectKeyExtraction<decltype(keyProjection), In1, In2, Rest...> (keyProjection, 0);

  // inject the key extraction into the predicate
  injectValueExtraction<decltype(valueProjection), In1, In2, Rest...> (valueProjection, 0);

  // the types for key and value
  using key = typename std::remove_reference<decltype(((Out*) nullptr)->getKey())>::type;
  using value = typename std::remove_reference<decltype(((Out*) nullptr)->getValue())>::type;

  // make sure the the type is fine
  static_assert(std::is_base_of<HandleBase, key>::value, "The key has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, value>::value, "The value has to be a Handle!");

  // make the join record lambda
  auto lambda = std::make_shared<JoinRecordLambda<Out, key, value>>(keyProjection.tree, valueProjection.tree);

  // create
  return LambdaTree<Handle<Out>>(lambda);
}

template <typename TypeToCallMethodOn, typename Out, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasFourArgs::testKeyProjection (&a)) *arg1 = nullptr, decltype (HasFourArgs::testValueProjection (&a)) *arg2 = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  GenericHandle fourth (4);

  // get the key and value projections
  auto valueProjection = a.getValueProjection (first, second, third, fourth);
  auto keyProjection = a.getKeyProjection (first, second, third, fourth);

  // inject the key extraction into the predicate
  injectKeyExtraction<decltype(keyProjection), In1, In2, Rest...> (keyProjection, 0);

  // inject the key extraction into the predicate
  injectValueExtraction<decltype(valueProjection), In1, In2, Rest...> (valueProjection, 0);

  // the types for key and value
  using key = typename std::remove_reference<decltype(((Out*) nullptr)->getKey())>::type;
  using value = typename std::remove_reference<decltype(((Out*) nullptr)->getValue())>::type;

  // make sure the the type is fine
  static_assert(std::is_base_of<HandleBase, key>::value, "The key has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, value>::value, "The value has to be a Handle!");

  // make the join record lambda
  auto lambda = std::make_shared<JoinRecordLambda<Out, key, value>>(keyProjection.tree, valueProjection.tree);

  // create
  return LambdaTree<Handle<Out>>(lambda);
}

template <typename TypeToCallMethodOn, typename Out, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasFiveArgs::testKeyProjection (&a)) *arg1 = nullptr, decltype (HasFiveArgs::testValueProjection (&a)) *arg2 = nullptr) {

  // make the input handles
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  GenericHandle fourth (4);
  GenericHandle fifth (5);

  // get the key and value projections
  auto valueProjection = a.getValueProjection (first, second, third, fourth, fifth);
  auto keyProjection = a.getKeyProjection (first, second, third, fourth, fifth);

  // inject the key extraction into the predicate
  injectKeyExtraction<decltype(keyProjection), In1, In2, Rest...> (keyProjection, 0);

  // inject the key extraction into the predicate
  injectValueExtraction<decltype(valueProjection), In1, In2, Rest...> (valueProjection, 0);

  // the types for key and value
  using key = typename std::remove_reference<decltype(((Out*) nullptr)->getKey())>::type;
  using value = typename std::remove_reference<decltype(((Out*) nullptr)->getValue())>::type;

  // make sure the the type is fine
  static_assert(std::is_base_of<HandleBase, key>::value, "The key has to be a Handle!");
  static_assert(std::is_base_of<HandleBase, value>::value, "The value has to be a Handle!");

  // make the join record lambda
  auto lambda = std::make_shared<JoinRecordLambda<Out, key, value>>(keyProjection.tree, valueProjection.tree);

  // create
  return LambdaTree<Handle<Out>>(lambda);
}

}
