#pragma once

// PRELOAD %SerConnectToRequest%

#include "Object.h"

namespace pdb {

class SerConnectToRequest : public Object {
public:

  SerConnectToRequest() = default;

  SerConnectToRequest(int32_t comp_id, int32_t job_id, int32_t node_id, int32_t taskID)
      : compID(comp_id), jobID(job_id), nodeID(node_id), taskID(taskID) {}

  // the id of the computation
  int32_t compID = -1;

  // the id of the job
  int32_t jobID = -1;

  // the id of the node
  int32_t nodeID = -1;

  // we use these to match tasks
  int32_t taskID = -1;

  ENABLE_DEEP_COPY

  bool operator==(const SerConnectToRequest &other) const{
    return (compID == other.compID && jobID == other.jobID &&
            nodeID == other.nodeID && taskID == other.taskID);
  }

};

class SerConnectToRequestHasher {
 public:

  // just do some silly hashing
  size_t operator()(const SerConnectToRequest& p) const
  {
    return std::hash<int32_t>()(p.compID) ^ std::hash<int32_t>()(p.jobID) ^
        std::hash<int32_t>()(p.nodeID) ^ std::hash<int32_t>()(p.taskID);
  }
};


}
