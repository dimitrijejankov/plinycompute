//
// Created by dimitrije on 4/14/19.
//

#ifndef PDB_JOINSIDE_H
#define PDB_JOINSIDE_H

#include <ComputeInfo.h>

namespace pdb {

class ShuffleJoinSide;
using ShuffleJoinSidePtr = std::shared_ptr<ShuffleJoinSide>;

/**
 * Specifies which side of the join we would do
 */
enum BroadcastJoinSideEnum {
  BUILD_SIDE,
  PROBE_SIDE
};

class BroadcastJoinSide : public ComputeInfo {
public:

  explicit BroadcastJoinSide(BroadcastJoinSideEnum value) : value(value) {}

  /**
   * The value of the enum
   */
  BroadcastJoinSideEnum value;

};


}


#endif //PDB_JOINSIDE_H
