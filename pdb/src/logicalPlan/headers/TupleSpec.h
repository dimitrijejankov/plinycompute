#include <utility>

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

#ifndef TUPLE_SPEC_H
#define TUPLE_SPEC_H

#include <iostream>
#include <memory>
#include <algorithm>
#include <stdlib.h>
#include <string>
#include <utility>
#include <vector>
#include <map>

#include "AttList.h"

// and here is the specifier for a TupleSet... it is basically a bunch of attribute
// names, as well as the name of the TupleSet
struct TupleSpec {

 private:
  std::string setName;
  std::vector<std::string> atts;

 public:
  TupleSpec() {
    setName = std::string("Empty");
  }

  ~TupleSpec() = default;

  explicit TupleSpec(std::string setNameIn) {
    setName = std::move(setNameIn);
  }

  TupleSpec(std::string setNameIn, AttList &useMe) {
    setName = std::move(setNameIn);
    atts = useMe.atts;
  }

  std::string &getSetName() {
    return setName;
  }

  void setSetName(const std::string &val) {
    setName = val;
  }

  std::vector<std::string> &getAtts() {
    return atts;
  }

  // removes the attribute from the attribute list
  void removeAtt(const std::string &att) {

    // remove the attribute
    atts.erase(std::remove(atts.begin(), atts.end(), att), atts.end());
  }

  // adds a new attribute to the tuple set
  void insertAtt(const std::string &att) {
    atts.emplace_back(att);
  }

  bool hasAtt(const std::string &att) {
    return std::find(atts.begin(), atts.end(), att) != atts.end();
  }

  // tries to find the attribute if we find it we replace it
  void replaceAtt(const std::string &replaceMe, const std::string replaceWithMe) {

    // try to find the attribute
    auto it = std::find(atts.begin(), atts.end(), replaceMe);

    // if found set it
    if(it != atts.end()) {
      *it = replaceWithMe;
    }
  }

  // keep all the attributes that are not in rhs for example lhs(a,b) \ rhs(b) = lhs(a)
  static TupleSpec complement(const TupleSpec &lhs, const TupleSpec &rhs) {

    // make the return value
    auto ret = TupleSpec(lhs.setName);

    // figure out the complements
    auto &rhsAtts = rhs.atts;
    for(const auto &att : lhs.atts) {

      // if the attribute is not in rhs keep it
      auto it = std::find(rhsAtts.begin(), rhsAtts.end(), att);
      if(it == rhsAtts.end()) {
        ret.getAtts().emplace_back(att);
      }
    }

    // return the value
    return ret;
  }

  bool isEmpty() {
    return setName == "Empty";
  }

  bool operator==(const TupleSpec &toMe) {
    return setName == toMe.setName;
  }

  friend std::ostream &operator<<(std::ostream &os, const TupleSpec &printMe);
};

inline std::ostream &operator<<(std::ostream &os, const TupleSpec &printMe) {
  os << printMe.setName << " (";
  bool first = true;
  for (auto &a : printMe.atts) {
    if (!first)
      os << ", ";
    first = false;
    os << a;
  }
  os << ")";
  return os;
}

#endif
