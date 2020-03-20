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

#include "Computation.h"
#include "PDBString.h"
#include "Object.h"
#include "LogicalPlan.h"
#include "PDBVector.h"
#include "pipeline/Pipeline.h"
#include "ComputeInfo.h"

namespace pdb {

// this is the basic type that is sent around a PDB cluster to store a computation that PDB is to execute
class ComputePlan {
 protected:

  // this data structure contains both the compiled TCAP string, as well as an index of all of the computations
  LogicalPlanPtr myPlan;

  // returns the source for a pipeline
  ComputeSourcePtr getComputeSource(int32_t nodeID,
                                    int32_t workerID,
                                    int32_t numWorkers,
                                    AtomicComputationPtr &sourceAtomicComputation,
                                    const PDBAbstractPageSetPtr &inputPageSet,
                                    std::map<ComputeInfoType, ComputeInfoPtr> &params);

  // returns the compute sink
  ComputeSinkPtr getComputeSink(AtomicComputationPtr &targetAtomicComp,
                                std::string& targetComputationName,
                                std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                size_t numNodes,
                                size_t numProcessingThreads);

  // assembles the pipeline with everything
  PipelinePtr assemblePipeline(const std::string &sourceTupleSetName,
                               const PDBAbstractPageSetPtr &sourcePageSet,
                               const PDBAbstractPageSetPtr &outputPageSet,
                               ComputeSourcePtr &computeSource,
                               ComputeSinkPtr &computeSink,
                               const PageProcessorPtr &processor,
                               std::map<ComputeInfoType, ComputeInfoPtr> &params,
                               std::vector<AtomicComputationPtr> &pipelineComputations,
                               size_t numNodes,
                               size_t numProcessingThreads,
                               uint64_t workerID);

  // return the result containing (targetSpec, targetAttsToOpOn, targetProjection)
  std::tuple<TupleSpec, TupleSpec, TupleSpec> getSinkSpecifier(AtomicComputationPtr &targetAtomicComp,
                                                               std::string &targetComputationName);

  // this does a DFS, trying to find a list of computations that lead to the specified computation
  static bool findPipelineComputations(const LogicalPlanPtr& myPlan,
                                       std::vector<AtomicComputationPtr> &listSoFar,
                                       const std::string &targetTupleSetName);

 public:

  ComputePlan() = default;

  explicit ComputePlan(LogicalPlanPtr myPlan);

  // returns the logical plan
  LogicalPlanPtr &getPlan();

  // builds a regular straight pipeline
  PipelinePtr buildPipeline(const std::string& sourceTupleSetName,
                            const std::string& targetTupleSetName,
                            const PDBAbstractPageSetPtr &inputPageSet,
                            const PDBAnonymousPageSetPtr &outputPageSet,
                            std::map<ComputeInfoType, ComputeInfoPtr> &params,
                            std::size_t nodeID,
                            size_t numNodes,
                            size_t numProcessingThreads,
                            uint64_t workerID);

  // build the aggregation pipeline
  PipelinePtr buildAggregationPipeline(const std::string &targetTupleSetName,
                                       const PDBAbstractPageSetPtr &inputPageSet,
                                       const PDBAnonymousPageSetPtr &outputPageSet,
                                       uint64_t workerID);

  // build a pipeline for the broadcast join
  PipelinePtr buildBroadcastJoinPipeline(const string &targetTupleSetName,
                                         const PDBAbstractPageSetPtr &inputPageSet,
                                         const PDBAnonymousPageSetPtr &outputPageSet,
                                         uint64_t numThreads,
                                         uint64_t numNodes,
                                         uint64_t workerID);

  // this will return the processor for the shuffle join
  PageProcessorPtr getProcessorForJoin(const std::string &joinTupleSetName,
                                       size_t numNodes,
                                       size_t numProcessingThreads,
                                       vector<PDBPageQueuePtr> &pageQueues,
                                       PDBBufferManagerInterfacePtr bufferManager);

};

}