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

#ifndef COMPUTE_PLAN_CC
#define COMPUTE_PLAN_CC

#include "ComputePlan.h"
#include "executors/FilterExecutor.h"
#include "AtomicComputationClasses.h"
#include "lambdas/EqualsLambda.h"
#include "JoinCompBase.h"
#include "AggregateCompBase.h"
#include "AggregationPipeline.h"
#include "JoinShufflePipeline.h"
#include "NullProcessor.h"
#include "Lexer.h"
#include "Parser.h"
#include "JoinComp.h"

extern int yydebug;

namespace pdb {

inline LogicalPlanPtr ComputePlan::getPlan() {

  // if we already have the plan, then just return it
  if (myPlan != nullptr)
    return myPlan;

  // get the string to compile
  std::string myLogicalPlan = TCAPComputation;
  myLogicalPlan.push_back('\0');

  // where the result of the parse goes
  AtomicComputationList *myResult;

  // now, do the compilation
  yyscan_t scanner;
  LexerExtra extra{""};
  yylex_init_extra(&extra, &scanner);
  const YY_BUFFER_STATE buffer{yy_scan_string(myLogicalPlan.data(), scanner)};
  const int parseFailed{yyparse(scanner, &myResult)};
  yy_delete_buffer(buffer, scanner);
  yylex_destroy(scanner);

  // if it didn't parse, get outta here
  if (parseFailed) {
    std::cout << "Parse error when compiling TCAP: " << extra.errorMessage;
    exit(1);
  }

  // this is the logical plan to return
  myPlan = std::make_shared<LogicalPlan>(*myResult, allComputations);
  delete myResult;

  // and now we are outta here
  return myPlan;
}

inline void ComputePlan::nullifyPlanPointer() {
  myPlan = nullptr;
}

// this does a DFS, trying to find a list of computations that lead to the specified computation
inline bool recurse(LogicalPlanPtr myPlan,
                    std::vector<AtomicComputationPtr> &listSoFar,
                    const std::string &targetTupleSetName) {

  // see if the guy at the end of the list is indeed the target
  if (listSoFar.back()->getOutputName() == targetTupleSetName) {

    // in this case, we have the complete list of computations
    return true;
  }

  // get all of the guys who consume the dude on the end of the list
  std::vector<AtomicComputationPtr>
      &nextOnes = myPlan->getComputations().getConsumingAtomicComputations(listSoFar.back()->getOutputName());

  // and try to put each of the next computations on the end of the list, and recursively search
  for (auto &a : nextOnes) {

    // see if the next computation was on the path to the target
    listSoFar.push_back(a);
    if (recurse(myPlan, listSoFar, targetTupleSetName)) {

      // it was!  So we are done
      return true;
    }

    // we couldn't find the target
    listSoFar.pop_back();
  }

  // if we made it here, we could not find the target
  return false;
}

inline PipelinePtr ComputePlan::buildPipeline(const std::string &sourceTupleSetName,
                                              const std::string &targetTupleSetName,
                                              const PDBAbstractPageSetPtr &inputPageSet,
                                              const PDBAnonymousPageSetPtr &outputPageSet,
                                              std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                              size_t numNodes,
                                              size_t numProcessingThreads,
                                              uint64_t chunkSize,
                                              uint64_t workerID) {

  // build the plan if it is not already done
  if (myPlan == nullptr)
    getPlan();

  // get all of the computations
  AtomicComputationList &allComps = myPlan->getComputations();

  std::cout << "print computations:" << std::endl;
  std::cout << allComps << std::endl;

  // get the atomic computation of the source
  auto sourceAtomicComputation = allComps.getProducingAtomicComputation(sourceTupleSetName);

  // now we get the name of the actual computation object that corresponds to the producer of this tuple set
  std::string producerName = sourceAtomicComputation->getComputationName();

  std::cout << "producerName = " << producerName << std::endl;

  // and get the schema for the output TupleSet objects that it is supposed to produce
  TupleSpec &origSpec = sourceAtomicComputation->getOutput();

  // now we are going to ask that particular node for the compute source
  ComputeSourcePtr computeSource = myPlan->getNode(producerName).getComputation().getComputeSource(inputPageSet, chunkSize, workerID);

  std::cout << "\nBUILDING PIPELINE\n";
  std::cout << "Source: " << origSpec << "\n";
  // now we have to do a DFS.  This vector will store all of the computations we've found so far
  std::vector<AtomicComputationPtr> listSoFar;

  // and this list stores the computations that we still need to process
  std::vector<AtomicComputationPtr> &nextOnes = myPlan->getComputations().getConsumingAtomicComputations(origSpec.getSetName());

  // now, see if each of the next guys can get us to the target tuple set
  bool gotIt = false;
  for (auto &a : nextOnes) {
    listSoFar.push_back(a);

    // see if the next computation was on the path to the target
    if (recurse(myPlan, listSoFar, targetTupleSetName)) {
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

  // and get the schema for the output TupleSet objects that it is supposed to produce
  TupleSpec &targetSpec = targetAtomicComp->getOutput();
  std::cout << "The target is " << targetSpec << "\n";

  // and get the projection for this guy
  std::vector<AtomicComputationPtr> &consumers = allComps.getConsumingAtomicComputations(targetSpec.getSetName());
  //JiaNote: change the reference into a new variable based on Chris' Join code
  //TupleSpec &targetProjection = targetSpec;
  TupleSpec targetProjection;
  TupleSpec targetAttsToOpOn;
  for (auto &a : consumers) {
    if (a->getComputationName() == targetComputationName) {

      std::cout << "targetComputationName was " << targetComputationName << "\n";

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
        std::cout << "This is bad... is the target computation name correct??";
        std::cout << "Didn't find a JoinSets, target was " << targetSpec.getSetName() << "\n";
        exit(1);
      }

      // get the join and make sure it matches
      auto *myGuy = (ApplyJoin *) a.get();
      if (!(myGuy->getRightInput() == targetSpec)) {
        std::cout << "This is bad... is the target computation name correct??";
        std::cout << "Find a JoinSets, target was " << targetSpec.getSetName() << "\n";
        exit(1);
      }

      std::cout << "Building sink for: " << targetSpec << " " << myGuy->getRightProjection() << " "
                << myGuy->getRightInput() << "\n";
      targetProjection = myGuy->getRightProjection();
      targetAttsToOpOn = myGuy->getRightInput();
      std::cout << "Building sink for: " << targetSpec << " " << targetAttsToOpOn << " " << targetProjection << "\n";
    }
  }

  // now we have the list of computations, and so it is time to build the pipeline... start by building a compute sink
  ComputeSinkPtr computeSink = myPlan->getNode(targetComputationName).getComputation().getComputeSink(targetSpec,
                                                                                                      targetAttsToOpOn,
                                                                                                      targetProjection,
                                                                                                      numProcessingThreads * numNodes,
                                                                                                      params,
                                                                                                      myPlan);

  // do we have a processor provided
  auto it = params.find(ComputeInfoType::PAGE_PROCESSOR);
  PageProcessorPtr processor = it != params.end() ? dynamic_pointer_cast<PageProcessor>(it->second) : make_shared<NullProcessor>();

  // make the pipeline
  std::shared_ptr<Pipeline> returnVal = std::make_shared<Pipeline>(outputPageSet, computeSource, computeSink, processor);

  // add the operations to the pipeline
  AtomicComputationPtr lastOne = myPlan->getComputations().getProducingAtomicComputation(sourceTupleSetName);
  for (auto &a : listSoFar) {

    // if we have a filter, then just go ahead and create it
    if (a->getAtomicComputationType() == "Filter") {

      // create a filter executor
      std::cout << "Adding: " << a->getProjection() << " + filter [" << a->getInput() << "] => " << a->getOutput() << "\n";
      returnVal->addStage(std::make_shared<FilterExecutor>(lastOne->getOutput(), a->getInput(), a->getProjection()));

      // if we had an apply, go ahead and find it and add it to the pipeline
    } else if (a->getAtomicComputationType() == "Apply") {

      // create an executor for the apply lambda
      std::cout << "Adding: " << a->getProjection() << " + apply [" << a->getInput() << "] => " << a->getOutput() << "\n";
      returnVal->addStage(myPlan->getNode(a->getComputationName()).
                          getLambda(((ApplyLambda *) a.get())->getLambdaToApply())->getExecutor(lastOne->getOutput(), a->getInput(), a->getProjection()));

    } else if (a->getAtomicComputationType() == "HashLeft") {

      // create an executor for left hasher
      std::cout << "Adding: " << a->getProjection() << " + hashleft [" << a->getInput() << "] => " << a->getOutput() << "\n";
      returnVal->addStage(myPlan->getNode(a->getComputationName()).
                          getLambda(((HashLeft *) a.get())->getLambdaToApply())->getLeftHasher(lastOne->getOutput(), a->getInput(), a->getProjection()));

    } else if (a->getAtomicComputationType() == "HashRight") {

      // create an executor for the right hasher
      std::cout << "Adding: " << a->getProjection() << " + hashright [" << a->getInput() << "] => " << a->getOutput() << "\n";
      returnVal->addStage(myPlan->getNode(a->getComputationName()).
                          getLambda(((HashLeft *) a.get())->getLambdaToApply())->getRightHasher(lastOne->getOutput(), a->getInput(), a->getProjection()));


    } else if (a->getAtomicComputationType() == "JoinSets") {

      std::cout << "Adding: " << a->getProjection() << " + join [" << a->getInput() << "] => " << a->getOutput() << "\n";

      // join is weird, because there are two inputs...
      auto &myComp = (JoinCompBase &) myPlan->getNode(a->getComputationName()).getComputation();
      auto *myJoin = (ApplyJoin *) (a.get());

      // grab the join arguments
      JoinArgumentsPtr joinArgs = std::dynamic_pointer_cast<JoinArguments>(params[ComputeInfoType::JOIN_ARGS]);
      if(joinArgs == nullptr) {
        throw runtime_error("Join pipeline run without hash tables!");
      }

      // do we have the appropriate join arguments? if not throw an exception
      auto it = joinArgs->hashTables.find(a->getOutput().getSetName());
      if(it != joinArgs->hashTables.end()) {
        throw runtime_error("Hash table for the output set," + a->getOutput().getSetName() +  "not found!");
      }

      // check if we are pipelining the right input
      if (lastOne->getOutput().getSetName() == myJoin->getRightInput().getSetName()) {

        // if we are pipelining the right input, then we don't need to switch left and right inputs
        std::cout << "We are pipelining the right input...\n";
        returnVal->addStage(myComp.getExecutor(true, myJoin->getProjection(), lastOne->getOutput(), myJoin->getRightInput(), myJoin->getRightProjection(), it->second, *this));

      } else {

        // if we are pipelining the right input, then we don't need to switch left and right inputs
        std::cout << "We are pipelining the left input...\n";
        returnVal->addStage(myComp.getExecutor(false, myJoin->getRightProjection(), lastOne->getOutput(), myJoin->getInput(), myJoin->getProjection(), it->second, *this));
      }

    }
    else if(a->getAtomicComputationType() == "WriteSet") {

      // skip this one
      std::cout << "We are skipping a write set this is essentially a NOOP\n";
    }
    else {
      std::cout << "This is bad... found an unexpected computation type (" << a->getComputationName()
                << ") inside of a pipeline.\n";
    }

    lastOne = a;
  }

  std::cout << "Sink: " << targetSpec << " [" << targetProjection << "]\n";
  return returnVal;
}

inline PipelinePtr ComputePlan::buildAggregationPipeline(const std::string &targetTupleSetName,
                                                         const PDBAbstractPageSetPtr &inputPageSet,
                                                         const PDBAnonymousPageSetPtr &outputPageSet,
                                                         uint64_t workerID) {

  // build the plan if it is not already done
  if (myPlan == nullptr)
    getPlan();

  // get all of the computations
  AtomicComputationList &allComps = myPlan->getComputations();

  // find the target atomic computation
  auto targetAtomicComp = allComps.getProducingAtomicComputation(targetTupleSetName);

  // find the target real PDBcomputation
  auto targetComputationName = targetAtomicComp->getComputationName();

  // grab the aggregation combiner
  Handle<AggregateCompBase> agg = unsafeCast<AggregateCompBase>(myPlan->getNode(targetComputationName).getComputationHandle());
  auto combiner = agg->getAggregationHashMapCombiner(workerID);

  return std::make_shared<pdb::AggregationPipeline>(workerID, outputPageSet, inputPageSet, combiner);
}

inline PipelinePtr ComputePlan::buildMergeJoinShufflePipeline(const string &targetTupleSetName,
                                                              const PDBAbstractPageSetPtr &inputPageSet,
                                                              const PDBAnonymousPageSetPtr &outputPageSet,
                                                              uint64_t numThreads,
                                                              uint64_t workerID) {

  // build the plan if it is not already done
  if (myPlan == nullptr)
    getPlan();

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

      std::cout << "targetComputationName was " << targetComputationName << "\n";

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
        std::cout << "This is bad... is the target computation name correct??";
        std::cout << "Didn't find a JoinSets, target was " << targetSpec.getSetName() << "\n";
        exit(1);
      }

      // get the join and make sure it matches
      auto *myGuy = (ApplyJoin *) a.get();
      if (!(myGuy->getRightInput() == targetSpec)) {
        std::cout << "This is bad... is the target computation name correct??";
        std::cout << "Find a JoinSets, target was " << targetSpec.getSetName() << "\n";
        exit(1);
      }

      std::cout << "Building sink for: " << targetSpec << " " << myGuy->getRightProjection() << " "
                << myGuy->getRightInput() << "\n";
      targetProjection = myGuy->getRightProjection();
      targetAttsToOpOn = myGuy->getRightInput();
      std::cout << "Building sink for: " << targetSpec << " " << targetAttsToOpOn << " " << targetProjection << "\n";
    }
  }

  Handle<JoinCompBase> joinComp = unsafeCast<JoinCompBase>(myPlan->getNode(targetComputationName).getComputationHandle());
  auto merger = joinComp->getComputeMerger(targetSpec, targetAttsToOpOn, targetProjection, workerID, numThreads, myPlan);

  return std::make_shared<pdb::JoinShufflePipeline>(workerID, outputPageSet, inputPageSet, merger);
}

inline ComputePlan::ComputePlan(String &TCAPComputation, Vector<Handle<Computation>> &allComputations) : TCAPComputation(TCAPComputation),
                                                                                                         allComputations(allComputations) {}


}

#endif


