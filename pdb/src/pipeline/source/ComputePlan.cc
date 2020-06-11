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

#include "ComputePlan.h"
#include "executors/FilterExecutor.h"
#include "executors/FlattenExecutor.h"
#include "executors/UnionExecutor.h"
#include "executors/HashOneExecutor.h"
#include "AtomicComputationClasses.h"
#include "lambdas/EqualsLambda.h"
#include "JoinCompBase.h"
#include "AggregateCompBase.h"
#include "AggregationPipeline.h"
#include "NullProcessor.h"
#include "Lexer.h"
#include "Parser.h"
#include "StringIntPair.h"
#include "JoinBroadcastPipeline.h"


extern int yydebug;


namespace pdb {

ComputePlan::ComputePlan(LogicalPlanPtr myPlan) : myPlan(std::move(myPlan)) {}

ComputeSourcePtr ComputePlan::getComputeSource(int32_t nodeID,
                                               int32_t workerID,
                                               int32_t numWorkers,
                                               AtomicComputationPtr &sourceAtomicComputation,
                                               const PDBAbstractPageSetPtr &inputPageSet,
                                               std::map<ComputeInfoType, ComputeInfoPtr> &params) {


  // now we get the name of the actual computation object that corresponds to the producer of this tuple set
  std::string producerName = sourceAtomicComputation->getComputationName();

  // get a reference to the computations of the logical plan
  auto &allComps = myPlan->getComputations();

  // if we are a join (shuffle join source) we need to have separate logic to handle that, otherwise just return a regular source
  if(sourceAtomicComputation->getAtomicComputationTypeID() != ApplyJoinTypeID) {

    // our source is a normal source and not a join source, so we just grab it from the computation
    return myPlan->getNode(producerName).getComputation().getComputeSource(inputPageSet, workerID, params);
  }

  // cast the join computation
  auto *joinComputation = (ApplyJoin *) sourceAtomicComputation.get();

  // grab the join arguments
  JoinArgumentsPtr joinArgs = std::dynamic_pointer_cast<JoinArguments>(params[ComputeInfoType::JOIN_ARGS]);
  if(joinArgs == nullptr) {
    throw runtime_error("Join pipeline run without hash tables!");
  }

  // is this a join aggregation algorithm we are running
  if(joinArgs->isJoinAggAggregation) {

    // get the right atomic computation
    AtomicComputationPtr rightAtomicComp = allComps.getProducingAtomicComputation(joinComputation->getRightInput().getSetName());

    // do we have the appropriate join arguments? if not throw an exception
    auto it = joinArgs->hashTables.find(rightAtomicComp->getOutput().getSetName());
    if(it == joinArgs->hashTables.end()) {
      throw runtime_error("Hash table for the output set," + rightAtomicComp->getOutput().getSetName() +  "not found!");
    }

    // get the left and right input page set
    auto leftInputPageSet = std::dynamic_pointer_cast<PDBRandomAccessPageSet>(inputPageSet);
    auto rightInputPageSet = std::dynamic_pointer_cast<PDBRandomAccessPageSet>(it->second->hashTablePageSet);

    // our source is a normal source and not a join source, so we just grab it from the computation
    return ((JoinCompBase *) &myPlan->getNode(joinComputation->getComputationName()).getComputation())->getJoinAggSource(nodeID,
                                                                                                                         workerID,
                                                                                                                         numWorkers,
                                                                                                                         *joinArgs->leftTIDToRecordMapping,
                                                                                                                         *joinArgs->rightTIDToRecordMapping,
                                                                                                                         joinArgs->planPage,
                                                                                                                         joinArgs->emitter,
                                                                                                                         leftInputPageSet,
                                                                                                                         rightInputPageSet);
  }

  // figure out if the source is the left or right side
  auto shuffleJoinArgs = std::dynamic_pointer_cast<ShuffleJoinArg>(params[ComputeInfoType::SHUFFLE_JOIN_ARG]);
  if(!shuffleJoinArgs->swapLeftAndRightSide) {

    AtomicComputationPtr leftAtomicComp = allComps.getProducingAtomicComputation(joinComputation->getInput().getSetName());
    AtomicComputationPtr rightAtomicComp = allComps.getProducingAtomicComputation(joinComputation->getRightInput().getSetName());
    bool needsToSwapSides = false;

    // do we have the appropriate join arguments? if not throw an exception
    auto it = joinArgs->hashTables.find(rightAtomicComp->getOutput().getSetName());
    if(it == joinArgs->hashTables.end()) {
      throw runtime_error("Hash table for the output set," + rightAtomicComp->getOutput().getSetName() +  "not found!");
    }

    // init the RHS source
    auto rhsSource = ((JoinCompBase *) &myPlan->getNode(joinComputation->getComputationName()).getComputation())->getRHSShuffleJoinSource(rightAtomicComp->getOutput(),
                                                                                                                                          joinComputation->getRightInput(),
                                                                                                                                          joinComputation->getRightProjection(),
                                                                                                                                          it->second->hashTablePageSet,
                                                                                                                                          myPlan,
                                                                                                                                          workerID);

    // init the compute source for the join
    return ((JoinCompBase *) &myPlan->getNode(joinComputation->getComputationName()).getComputation())->getJoinedSource(joinComputation->getProjection(), // this tells me how the join tuple of the LHS is layed out
                                                                                                                        rightAtomicComp->getOutput(), // this gives the specification of the RHS tuple
                                                                                                                        joinComputation->getRightInput(), // this gives the location of the RHS hash
                                                                                                                        joinComputation->getRightProjection(), // this gives the projection of the RHS tuple
                                                                                                                        rhsSource, // the RHS source that gives us the tuples
                                                                                                                        inputPageSet, // the LHS page set
                                                                                                                        myPlan,
                                                                                                                        needsToSwapSides,
                                                                                                                        workerID);

  }
  else {

    AtomicComputationPtr rightAtomicComp = allComps.getProducingAtomicComputation(joinComputation->getInput().getSetName());
    AtomicComputationPtr leftAtomicComp = allComps.getProducingAtomicComputation(joinComputation->getRightInput().getSetName());
    bool needsToSwapSides = true;

    // do we have the appropriate join arguments? if not throw an exception
    auto it = joinArgs->hashTables.find(rightAtomicComp->getOutput().getSetName());
    if(it == joinArgs->hashTables.end()) {
      throw runtime_error("Hash table for the output set," + rightAtomicComp->getOutput().getSetName() +  " not found!");
    }

    // init the RHS source
    auto rhsSource = ((JoinCompBase *) &myPlan->getNode(joinComputation->getComputationName()).getComputation())->getRHSShuffleJoinSource(rightAtomicComp->getOutput(),
                                                                                                                                          joinComputation->getInput(),
                                                                                                                                          joinComputation->getProjection(),
                                                                                                                                          it->second->hashTablePageSet,
                                                                                                                                          myPlan,
                                                                                                                                          workerID);

    // init the compute source for the join
    return ((JoinCompBase *) &myPlan->getNode(joinComputation->getComputationName()).getComputation())->getJoinedSource(joinComputation->getRightProjection(), // this tells me how the join tuple of the LHS is layed out
                                                                                                                        rightAtomicComp->getOutput(), // this gives the specification of the RHS tuple
                                                                                                                        joinComputation->getInput(), // this gives the location of the RHS hash
                                                                                                                        joinComputation->getProjection(), // this gives the projection of the RHS tuple
                                                                                                                        rhsSource, // the RHS source that gives us the tuples
                                                                                                                        inputPageSet, // the LHS page set
                                                                                                                        myPlan,
                                                                                                                        needsToSwapSides,
                                                                                                                        workerID);
  }
}

std::tuple<TupleSpec, TupleSpec, TupleSpec> ComputePlan::getSinkSpecifier(AtomicComputationPtr &targetAtomicComp,
                                                                          std::string &targetComputationName) {

  // get a reference to the computations of the logical plan
  auto &allComps = myPlan->getComputations();

  // and get the schema for the output TupleSet objects that it is supposed to produce
  TupleSpec &targetSpec = targetAtomicComp->getOutput();

  // and get the projection for this guy
  const auto &consumers = allComps.getConsumingAtomicComputations(targetSpec.getSetName());

  /// TODO this whole part needs to be rewritten
  TupleSpec targetProjection = targetSpec;
  TupleSpec targetAttsToOpOn = targetSpec;
  for (auto &a : consumers) {
    if (a->getComputationName() == targetComputationName) {

      // we found the consuming computation
      if (targetSpec == a->getInput()) {
        targetProjection = a->getProjection();

        //added following to merge join code
        if (targetComputationName.find("JoinComp") == std::string::npos) {
          targetSpec = targetProjection;
        }

        targetAttsToOpOn = a->getInput();
        break;
      }

      // the only way that the input to this guy does not match targetSpec is if he is a join, which has two inputs
      if (a->getAtomicComputationType() != std::string("JoinSets")) {
        exit(1);
      }

      // get the join and make sure it matches
      auto *myGuy = (ApplyJoin *) a.get();
      if (!(myGuy->getRightInput() == targetSpec)) {
        exit(1);
      }

      targetProjection = myGuy->getRightProjection();
      targetAttsToOpOn = myGuy->getRightInput();
    }
  }

  // return the result containing (targetSpec, targetAttsToOpOn, targetProjection)
  return std::move(make_tuple(targetSpec, targetAttsToOpOn, targetProjection));
}

ComputeSinkPtr ComputePlan::getComputeSink(AtomicComputationPtr &targetAtomicComp,
                                           std::string& targetComputationName,
                                           std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                           size_t numNodes,
                                           size_t numProcessingThreads) {


  // returns the input specifier
  auto specifier = getSinkSpecifier(targetAtomicComp, targetComputationName);

  // now we have the list of computations, and so it is time to build the pipeline... start by building a compute sink
  return myPlan->getNode(targetComputationName).getComputation().getComputeSink(std::get<0>(specifier),
                                                                                std::get<1>(specifier),
                                                                                std::get<2>(specifier),
                                                                                numNodes,
                                                                                numProcessingThreads,
                                                                                params,
                                                                                myPlan);
}

PipelinePtr ComputePlan::assemblePipeline(const std::string& sourceTupleSetName,
                                          const PDBAbstractPageSetPtr &outputPageSet,
                                          ComputeSourcePtr &computeSource,
                                          ComputeSinkPtr &computeSink,
                                          const PageProcessorPtr &processor,
                                          std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                          std::vector<AtomicComputationPtr> &pipelineComputations,
                                          size_t numNodes,
                                          size_t numProcessingThreads,
                                          uint64_t workerID) {

  // make the pipeline
  std::shared_ptr<Pipeline> returnVal = std::make_shared<Pipeline>(outputPageSet, computeSource, computeSink, processor);

  // add the operations to the pipeline
  AtomicComputationPtr lastOne = myPlan->getComputations().getProducingAtomicComputation(sourceTupleSetName);
  for (auto &a : pipelineComputations) {

    // if we have a filter, then just go ahead and create it
    if (a->getAtomicComputationType() == "Filter") {

      // create a filter executor
      returnVal->addStage(std::make_shared<FilterExecutor>(lastOne->getOutput(), a->getInput(), a->getProjection()));

      // if we had an apply, go ahead and find it and add it to the pipeline
    } else if (a->getAtomicComputationType() == "Apply") {

      // create an executor for the apply lambda
      returnVal->addStage(myPlan->getNode(a->getComputationName()).
          getLambda(((ApplyLambda *) a.get())->getLambdaToApply())->getExecutor(lastOne->getOutput(), a->getInput(), a->getProjection()));

    } else if(a->getAtomicComputationType() == "Union") {

      // get the union
      auto u = (Union *) a.get();

      // check if we are pipelining the right input
      if (lastOne->getOutput().getSetName() == u->getRightInput().getSetName()) {

        returnVal->addStage(std::make_shared<UnionExecutor>(lastOne->getOutput(), u->getRightInput()));

      } else {

        returnVal->addStage(std::make_shared<UnionExecutor>(lastOne->getOutput(), u->getInput()));
      }

    } else if (a->getAtomicComputationType() == "HashLeft") {

      // create an executor for left hasher
      returnVal->addStage(myPlan->getNode(a->getComputationName()).
          getLambda(((HashLeft *) a.get())->getLambdaToApply())->getLeftHasher(lastOne->getOutput(), a->getInput(), a->getProjection()));

    } else if (a->getAtomicComputationType() == "HashRight") {

      // create an executor for the right hasher
      returnVal->addStage(myPlan->getNode(a->getComputationName()).
          getLambda(((HashLeft *) a.get())->getLambdaToApply())->getRightHasher(lastOne->getOutput(), a->getInput(), a->getProjection()));


    } else if (a->getAtomicComputationType() == "HashOne") {

      returnVal->addStage(std::make_shared<HashOneExecutor>(lastOne->getOutput(), a->getInput(), a->getProjection()));

    } else if (a->getAtomicComputationType() == "Flatten") {

      returnVal->addStage(std::make_shared<FlattenExecutor>(lastOne->getOutput(), a->getInput(), a->getProjection()));

    } else if (a->getAtomicComputationType() == "JoinSets") {

      // join is weird, because there are two inputs...
      auto &myComp = (JoinCompBase &) myPlan->getNode(a->getComputationName()).getComputation();
      auto *myJoin = (ApplyJoin *) (a.get());

      // grab the join arguments
      JoinArgumentsPtr joinArgs = std::dynamic_pointer_cast<JoinArguments>(params[ComputeInfoType::JOIN_ARGS]);
      if(joinArgs == nullptr) {
        throw runtime_error("Join pipeline run without hash tables!");
      }

      // init the parameters
      bool needToSwapAtts = lastOne->getOutput().getSetName() == myJoin->getRightInput().getSetName();
      TupleSpec hashedInputSchema = needToSwapAtts ? myJoin->getProjection() : myJoin->getRightProjection();
      TupleSpec pipelinedInputSchema = lastOne->getOutput();
      TupleSpec pipelinedAttsToOperateOn = needToSwapAtts ? myJoin->getRightInput() : myJoin->getInput();
      TupleSpec pipelinedAttsToIncludeInOutput = needToSwapAtts ? myJoin->getRightProjection() : myJoin->getProjection();

      // do we have the appropriate join arguments? if not throw an exception
      auto it = joinArgs->hashTables.find(needToSwapAtts ? myJoin->getInput().getSetName() : myJoin->getRightInput().getSetName());
      if (it == joinArgs->hashTables.end()) {
        throw runtime_error("Hash table for the output set," + a->getOutput().getSetName() + "not found!");
      }

      // if this is a keyed join this is bad.
      if(myJoin->isKeyJoin) {
        throw runtime_error("The join is keyed for " + a->getOutput().getSetName() + " can not add an executor for that!");
      }

      // if we are pipelining the right input, then we don't need to switch left and right inputs
      returnVal->addStage(myComp.getExecutor(needToSwapAtts,
                                                 hashedInputSchema,
                                                 pipelinedInputSchema,
                                                 pipelinedAttsToOperateOn,
                                                 pipelinedAttsToIncludeInOutput,
                                                 it->second,
                                                     numNodes,
                                                     numProcessingThreads,
                                                     workerID,
                                                 *this));
    }
    else if(a->getAtomicComputationType() == "WriteSet") {

      // skip this one
      std::cout << "We are skipping a write set this is essentially a NOOP\n";
    }
    else {
      std::cout << "This is bad... found an unexpected computation type (" << a->getAtomicComputationType()
                << ") inside of a pipeline.\n";
    }
    lastOne = a;
  }

  return std::move(returnVal);
}

bool ComputePlan::findPipelineComputations(const LogicalPlanPtr& myPlan,
                                           std::vector<AtomicComputationPtr> &listSoFar,
                                           const std::string &targetTupleSetName) {

  // see if the guy at the end of the list is indeed the target
  if (listSoFar.back()->getOutputName() == targetTupleSetName) {

    // in this case, we have the complete list of computations
    return true;
  }

  // get all of the guys who consume the dude on the end of the list
  std::vector<AtomicComputationPtr> &nextOnes = myPlan->getComputations().getConsumingAtomicComputations(listSoFar.back()->getOutputName());

  // and try to put each of the next computations on the end of the list, and recursively search
  for (auto &a : nextOnes) {

    // see if the next computation was on the path to the target
    listSoFar.push_back(a);
    if (findPipelineComputations(myPlan, listSoFar, targetTupleSetName)) {

      // it was!  So we are done
      return true;
    }

    // we couldn't find the target
    listSoFar.pop_back();
  }

  // if we made it here, we could not find the target
  return false;
}

LogicalPlanPtr &ComputePlan::getPlan() {
  return myPlan;
}

PipelinePtr ComputePlan::buildPipeline(const std::string& sourceTupleSetName,
                                       const std::string& targetTupleSetName,
                                       const PDBAbstractPageSetPtr &inputPageSet,
                                       const PDBAnonymousPageSetPtr &outputPageSet,
                                       std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                       std::size_t nodeID,
                                       size_t numNodes,
                                       size_t numProcessingThreads,
                                       uint64_t workerID) {

  // get all of the computations
  AtomicComputationList &allComps = myPlan->getComputations();

  /// 0. Figure out the compute source

  // get the atomic computation of the source
  auto sourceAtomicComputation = allComps.getProducingAtomicComputation(sourceTupleSetName);

  // and get the schema for the output TupleSet objects that it is supposed to produce
  TupleSpec &origSpec = sourceAtomicComputation->getOutput();

  // figure out the source
  ComputeSourcePtr computeSource = getComputeSource(nodeID, workerID, numProcessingThreads, sourceAtomicComputation, inputPageSet, params);

  /// 1. Find all the computations in the pipeline

  // now we have to do a DFS.  This vector will store all of the computations we've found so far
  std::vector<AtomicComputationPtr> listSoFar;

  // and this list stores the computations that we still need to process
  std::vector<AtomicComputationPtr> &nextOnes = myPlan->getComputations().getConsumingAtomicComputations(origSpec.getSetName());

  // now, see if each of the next guys can get us to the target tuple set
  bool gotIt = false;
  for (auto &a : nextOnes) {
    listSoFar.push_back(a);

    // see if the next computation was on the path to the target
    if (findPipelineComputations(myPlan, listSoFar, targetTupleSetName)) {
      gotIt = true;
      break;
    }

    // we couldn't find the target
    listSoFar.pop_back();
  }

  // see if we could not find a path
  if (!gotIt) {
    std::cerr << "This is bad.  Could not find a path from source computation to sink computation.\n";
    exit(1);
  }

  /// 2. Figure out the sink

  // find the target atomic computation
  auto targetAtomicComp = allComps.getProducingAtomicComputation(targetTupleSetName);

  // find the target real PDBComputation
  auto targetComputationName = targetAtomicComp->getComputationName();

  // if the write set is in the pipeline remove it since it is basically a noop
  if(targetAtomicComp->getAtomicComputationType() == "WriteSet") {

    // pop it!
    listSoFar.pop_back();
    targetAtomicComp = listSoFar.back();
  }

  // get the compute sink
  auto computeSink = getComputeSink(targetAtomicComp, targetComputationName, params, numNodes, numProcessingThreads);

  // do we have a processor provided
  auto it = params.find(ComputeInfoType::PAGE_PROCESSOR);
  PageProcessorPtr processor = it != params.end() ? dynamic_pointer_cast<PageProcessor>(it->second) : make_shared<NullProcessor>();

  /// 3. Assemble the pipeline

  // assemble the whole pipeline
  return std::move(assemblePipeline(sourceTupleSetName,
                                    outputPageSet,
                                    computeSource,
                                    computeSink,
                                    processor,
                                    params,
                                    listSoFar,
                                    numNodes,
                                    numProcessingThreads,
                                    workerID));
}


PipelinePtr ComputePlan::buildAggregationPipeline(const std::string &targetTupleSetName,
                                                  const PDBWorkerQueuePtr &workerQueue,
                                                  const PDBAbstractPageSetPtr &inputPageSet,
                                                  const PDBAnonymousPageSetPtr &outputPageSet,
                                                  uint64_t workerID) {

  // get all of the computations
  AtomicComputationList &allComps = myPlan->getComputations();

  // find the target atomic computation
  auto targetAtomicComp = allComps.getProducingAtomicComputation(targetTupleSetName);

  // find the target real PDBComputation
  auto targetComputationName = targetAtomicComp->getComputationName();

  // grab the aggregation combiner
  Handle<AggregateCompBase> agg = unsafeCast<AggregateCompBase>(myPlan->getNode(targetComputationName).getComputationHandle());
  auto combiner = agg->getAggregationHashMapCombiner(workerID);

  return std::make_shared<pdb::AggregationPipeline>(workerID, outputPageSet, inputPageSet, workerQueue, combiner);
}


PipelinePtr ComputePlan::buildBroadcastJoinPipeline(const string &targetTupleSetName,
                                                           const PDBAbstractPageSetPtr &inputPageSet,
                                                           const PDBAnonymousPageSetPtr &outputPageSet,
                                                           uint64_t numThreads,
                                                           uint64_t numNodes,
                                                           uint64_t workerID) {

  // get all of the computations
  AtomicComputationList &allComps = myPlan->getComputations();

  // find the target atomic computation
  auto targetAtomicComp = allComps.getProducingAtomicComputation(targetTupleSetName);

  // find the target real PDBComputation
  auto targetComputationName = targetAtomicComp->getComputationName();

  // and get the schema for the output TupleSet objects that it is supposed to produce
  TupleSpec &targetSpec = targetAtomicComp->getOutput();

  // and get the projection for this guy
  std::vector<AtomicComputationPtr> &consumers = allComps.getConsumingAtomicComputations(targetSpec.getSetName());

  TupleSpec targetProjection;
  TupleSpec targetAttsToOpOn;
  for (auto &a : consumers) {
    if (a->getComputationName() == targetComputationName) {

      // we found the consuming computation
      if (targetSpec == a->getInput()) {
        targetProjection = a->getProjection();

        //added following to merge join code
        if (targetComputationName.find("JoinComp") == std::string::npos) {
          targetSpec = targetProjection;
        }

        targetAttsToOpOn = a->getInput();
        break;
      }

      // the only way that the input to this guy does not match targetSpec is if he is a join, which has two inputs
      if (a->getAtomicComputationType() != std::string("JoinSets")) {
        exit(1);
      }

      // get the join and make sure it matches
      auto *myGuy = (ApplyJoin *) a.get();
      if (!(myGuy->getRightInput() == targetSpec)) {
        exit(1);
      }

      targetProjection = myGuy->getRightProjection();
      targetAttsToOpOn = myGuy->getRightInput();
    }
  }

  // get the join computation
  Handle<JoinCompBase> joinComp = unsafeCast<JoinCompBase>(myPlan->getNode(targetComputationName).getComputationHandle());

  // get the BroadcastJoin pipeline merger
  auto merger = joinComp->getComputeMerger(targetSpec, targetAttsToOpOn, targetProjection, workerID, numThreads, numNodes, myPlan);

  // build the BroadcastJoin pipelines
  return std::make_shared<pdb::JoinBroadcastPipeline>(workerID, outputPageSet, inputPageSet, merger);
}

PageProcessorPtr ComputePlan::getProcessorForJoin(const std::string &tupleSetName,
                                                  size_t numNodes,
                                                  size_t numProcessingThreads,
                                                  vector<PDBPageQueuePtr> &pageQueues,
                                                  PDBBufferManagerInterfacePtr bufferManager) {

  // get all of the computations
  AtomicComputationList &allComps = myPlan->getComputations();

  //
  auto joinComputation = allComps.getProducingAtomicComputation(tupleSetName);
  TupleSpec &targetSpec = joinComputation->getOutput();

  // find the target atomic computation
  std::vector<AtomicComputationPtr> &consumers = allComps.getConsumingAtomicComputations(targetSpec.getSetName());

  //
  TupleSpec targetProjection;
  for (auto &a : consumers) {

    //
    if (targetSpec == a->getInput()) {

      // get the projection
      targetProjection = a->getProjection();
      break;
    }

    // get the join and make sure it matches
    auto *myGuy = (ApplyJoin *) a.get();
    if (!(myGuy->getRightInput() == targetSpec)) {
      throw runtime_error("");
    }

    targetProjection = myGuy->getRightProjection();
  }

  // return the processor
  return  ((JoinCompBase*) &myPlan->getNode(joinComputation->getComputationName()).getComputation())->getShuffleJoinProcessor(numNodes,
                                                                                                                              numProcessingThreads,
                                                                                                                              pageQueues,
                                                                                                                              bufferManager,
                                                                                                                              targetProjection,
                                                                                                                              myPlan);
}

}