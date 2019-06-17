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
#ifndef JOINCOMPBASE_H
#define JOINCOMPBASE_H

#include "Computation.h"
#include "RHSShuffleJoinSourceBase.h"
#include <JoinArguments.h>

namespace pdb {

class JoinCompBase : public Computation {

 public:

  virtual ComputeExecutorPtr getExecutor(bool needToSwapAtts,
                                         TupleSpec &hashedInputSchema,
                                         TupleSpec &pipelinedInputSchema,
                                         TupleSpec &pipelinedAttsToOperateOn,
                                         TupleSpec &pipelinedAttsToIncludeInOutput,
                                         JoinArgPtr &joinArg,
                                         uint64_t numNodes,
                                         uint64_t numProcessingThreads,
                                         uint64_t workerID,
                                         ComputePlan &computePlan) = 0;

  virtual ComputeExecutorPtr getExecutor(bool needToSwapAtts,
                                         TupleSpec &hashedInputSchema,
                                         TupleSpec &pipelinedInputSchema,
                                         TupleSpec &pipelinedAttsToOperateOn,
                                         TupleSpec &pipelinedAttsToIncludeInOutput) = 0;

  virtual RHSShuffleJoinSourceBasePtr getRHSShuffleJoinSource(TupleSpec &inputSchema,
                                                              TupleSpec &hashSchema,
                                                              TupleSpec &recordSchema,
                                                              const PDBAbstractPageSetPtr &leftInputPageSet,
                                                              pdb::LogicalPlanPtr &plan,
                                                              uint64_t chunkSize,
                                                              uint64_t workerID) = 0;

  virtual ComputeSourcePtr getJoinedSource(TupleSpec &outputSchema,
                                           TupleSpec &inputSchemaRHS,
                                           TupleSpec &hashSchemaRHS,
                                           TupleSpec &recordSchemaRHS,
                                           RHSShuffleJoinSourceBasePtr leftSource,
                                           const PDBAbstractPageSetPtr &rightInputPageSet,
                                           pdb::LogicalPlanPtr &plan,
                                           bool needToSwapLHSAndRhs,
                                           uint64_t chunkSize,
                                           uint64_t workerID) = 0;

  virtual PageProcessorPtr getShuffleJoinProcessor(size_t numNodes,
                                                   size_t numProcessingThreads,
                                                   vector<PDBPageQueuePtr> &pageQueues,
                                                   PDBBufferManagerInterfacePtr &bufferManager,
                                                   TupleSpec &recordSchema,
                                                   pdb::LogicalPlanPtr &plan) = 0;

  virtual ComputeSinkPtr getComputeMerger(TupleSpec &consumeMe,
                                          TupleSpec &attsToOpOn,
                                          TupleSpec &projection,
                                          uint64_t workerID,
                                          uint64_t numThreads,
                                          uint64_t numNodes,
                                          pdb::LogicalPlanPtr &plan) = 0;

};

}

#endif
