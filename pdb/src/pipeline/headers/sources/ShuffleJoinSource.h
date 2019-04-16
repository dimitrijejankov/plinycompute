//
// Created by dimitrije on 4/16/19.
//

#pragma once

#include <ComputeSource.h>
#include <PDBPageHandle.h>
#include <JoinMap.h>

namespace pdb {

/**
 *
 * @tparam LHS
 */
template<typename LHS>
class LeftShuffleJoinSource : public ComputeSource {
public:

  // the current page
  PDBPageHandle currPage;

  // the current join map
  Handle<JoinMap<LHS>> currMap;

  // the iterator
  JoinMapIterator<LHS> currIterator;

  LeftShuffleJoinSource() {

  }

  TupleSetPtr getNextTupleSet() override {



    return nullptr;
  };

};

/**
 *
 * @tparam RHS
 */
template<typename RHS>
class JoinedShuffleJoinSource : public ComputeSource {
 public:

  TupleSetPtr getNextTupleSet() override {
    return nullptr;
  };
};

}
