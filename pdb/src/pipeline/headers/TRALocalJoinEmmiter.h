#pragma once

#include "JoinAggTupleEmitter.h"

namespace pdb {

class TRALocalJoinEmmiter : public JoinAggTupleEmitterInterface {

  void getRecords(std::vector<JoinedRecord> &putHere,
                  int32_t &lastLHSPage,
                  int32_t &lastRHSPage,
                  int32_t threadID) override {}
};

}