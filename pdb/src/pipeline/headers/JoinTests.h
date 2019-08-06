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

#ifndef JOIN_TESTS_H
#define JOIN_TESTS_H

#include "Handle.h"

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
};

/**
 *
 */

template <typename LambdaType, typename In1, typename ...Rest>
typename std::enable_if<sizeof ...(Rest) != 0, void>::type
injectIntoSelection(LambdaType predicate, int input) {

  injectIntoSelection<LambdaType, Rest...>(predicate, input + 1);

  // prepare the input
  GenericHandle tmp(input + 1);
  Handle<In1> in = tmp;

  // inject the key lambda
  predicate.inject(input, LambdaTree<Ptr<In1>>(std::make_shared<KeyExtractionLambda<In1>>(in)));
}

template <typename LambdaType, typename In1>
void injectIntoSelection(LambdaType predicate, int input) {

  // prepare the input
  GenericHandle tmp(input + 1);
  Handle<In1> in = tmp;

  // inject the key lambda
  predicate.inject(input, LambdaTree<Ptr<In1>>(std::make_shared<KeyExtractionLambda<In1>>(in)));
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
  injectIntoSelection<decltype(predicate), In1, In2, Rest...> (predicate, 0);

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
  auto predicate = a.getKeySelection (first, second);

  // inject the key extraction into the predicate
  injectIntoSelection<decltype(predicate), In1, In2, Rest...> (predicate, 0);

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
  auto predicate = a.getKeySelection (first, second);

  // inject the key extraction into the predicate
  injectIntoSelection<decltype(predicate), In1, In2, Rest...> (predicate, 0);

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
  auto predicate = a.getKeySelection (first, second);

  // inject the key extraction into the predicate
  injectIntoSelection<decltype(predicate), In1, In2, Rest...> (predicate, 0);

  // return the predicate
  return predicate;
}

/**
 *
 */

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasTwoArgs::testProjection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  return a.getProjection (first, second);
}

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasThreeArgs::testProjection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  return a.getProjection (first, second, third);
}

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasFourArgs::testProjection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  GenericHandle fourth (4);
  return a.getProjection (first, second, third, fourth);
}

template <typename TypeToCallMethodOn, typename In1, typename In2, typename ...Rest>
auto callGetProjection (TypeToCallMethodOn &a, decltype (HasFiveArgs::testProjection (&a)) *arg = nullptr) {
  GenericHandle first (1);
  GenericHandle second (2);
  GenericHandle third (3);
  GenericHandle fourth (4);
  GenericHandle fifth (5);
  return a.getProjection (first, second, third, fourth, fifth);
}

}

#endif
