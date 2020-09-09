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

#ifndef JOIN_COMP
#define JOIN_COMP

#include <JoinTupleSingleton.h>
#include <lambdas/KeyExtractionLambda.h>
#include <JoinMapCreator.h>
#include "Computation.h"
#include "JoinTests.h"
#include "ComputePlan.h"
#include "JoinTuple.h"
#include "SelfLambda.h"
#include "JoinCompBase.h"
#include "LogicalPlan.h"
#include "MultiInputsBase.h"
#include "TRAJoinSource.h"
#include "JoinAggSource.h"
#include "Lambda.h"

namespace pdb {

template<typename Derived, typename Out, typename In1, typename In2, typename ...Rest>
class JoinComp : public JoinCompBase {
 public:

  virtual ~JoinComp() = default;

  // calls getProjection and getSelection to extract the lambdas
  void extractLambdas(std::map<std::string, LambdaObjectPtr> &returnVal) override {

    // get the selection lambda
    Lambda<bool> selectionLambda = callGetSelection<Derived, In1, In2, Rest...>(*static_cast<Derived *>(this));
    Lambda<Handle<Out>>
        projectionLambda = callGetProjection<Derived, Out, In1, In2, Rest...>(*static_cast<Derived *>(this));

    // the label we are started labeling
    int32_t startLabel = 0;

    // extract the lambdas
    selectionLambda.extractLambdas(returnVal, startLabel);
    projectionLambda.extractLambdas(returnVal, startLabel);
  }

  // returns the key extractor for the materialized result of this
  PDBKeyExtractorPtr getKeyExtractor() override {

    // check if we can extract a key
    constexpr bool hasGetKey = pdb::tupleTests::has_get_key<Out>::value;

    // get the key
    if constexpr (hasGetKey) {
      return std::make_shared<pdb::PDBKeyExtractorImpl<Out>>();
    }

    // we can not do this
    return nullptr;
  }

  // return the output type
  std::string getOutputType() override {
    return getTypeName<Out>();
  }

  // is the key
  bool isKeyed() override {
    return joinUsesKey<Derived>(*static_cast<Derived *>(this));
  }

  // count the number of inputs
  int getNumInputs() final {
    const int extras = sizeof...(Rest);
    return extras + 2;
  }

  template<typename First, typename ...Others>
  typename std::enable_if<sizeof ...(Others) == 0, std::string>::type getInputType(int i) {
    if (i == 0) {
      return getTypeName<First>();
    } else {
      std::cout << "Asked for an input type that didn't exist!";
      exit(1);
    }
  }

  // helper function to get a particular intput type
  template<typename First, typename ...Others>
  typename std::enable_if<sizeof ...(Others) != 0, std::string>::type getInputType(int i) {
    if (i == 0) {
      return getTypeName<First>();
    } else {
      return getInputType<Others...>(i - 1);
    }
  }

  // from the interface: get the i^th input type
  std::string getInputType(int i) final {
    return getInputType<In1, In2, Rest...>(i);
  }

  JoinAggSideSenderPtr getJoinAggSender(TupleSpec &projection,
                                        pdb::LogicalPlanPtr &plan,
                                        PDBPageHandle page,
                                        PDBCommunicatorPtr comm) override {

    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(projection, plan, whereEveryoneGoes);

    // check what kind of sink we need
    return correctJoinTuple->getJoinAggSender(page, comm);
  }

  JoinMapCreatorPtr getJoinMapCreator(TupleSpec &projection,
                                      pdb::LogicalPlanPtr &logicalPlan,
                                      PDBRandomAccessPageSetPtr pageSet,
                                      PDBCommunicatorPtr communicator,
                                      const JoinAggTupleEmitterPtr &emitter,
                                      bool isLHS,
                                      PDBLoggerPtr logger) override {

    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(projection, logicalPlan, whereEveryoneGoes);

    // check what kind of sink we need
    return correctJoinTuple->getJoinMapCreator(pageSet,
                                               communicator,
                                               emitter,
                                               isLHS,
                                               logger);
  }

  // this gets a compute sink
  ComputeSinkPtr getComputeSink(TupleSpec &consumeMe,
                                TupleSpec &attsToOpOn,
                                TupleSpec &projection,
                                uint64_t numNodes,
                                uint64_t numThreads,
                                std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                pdb::LogicalPlanPtr &plan) override {

    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(projection, plan, whereEveryoneGoes);

    // try to find the join agg arguments
    auto it = params.find(ComputeInfoType::JOIN_ARGS);
    if (it == params.end()) {
      throw runtime_error("Could not find the join arguments");
    }

    // the join arguments
    JoinArgumentsPtr joinArgs = dynamic_pointer_cast<JoinArguments>(it->second);

    // check what kind of sink we need
    if (!joinArgs->isJoinAggSide) {
      return correctJoinTuple->getSink(consumeMe, attsToOpOn, projection, whereEveryoneGoes, numNodes * numThreads);
    } else {

      // could not find the
      it = params.find(ComputeInfoType::JOIN_AGG_SIDE_ARGS);
      if (it == params.end()) {
        throw runtime_error("Could not find the join aggregation side arguments");
      }

      // the join arguments
      JoinAggSideArgPtr joinSideArgs = dynamic_pointer_cast<JoinAggSideArg>(it->second);

      return correctJoinTuple->getJoinAggSink(consumeMe, attsToOpOn, projection, whereEveryoneGoes,
                                              joinSideArgs->keyToNode, joinSideArgs->senders, joinSideArgs->plan,
                                              joinSideArgs->isLeft, numNodes * numThreads);
    }
  }

  // this gets the key sink
  ComputeSinkPtr getKeySink(TupleSpec &consumeMe,
                            TupleSpec &attsToOpOn,
                            TupleSpec &projection,
                            uint64_t numPartitions,
                            std::map<ComputeInfoType, ComputeInfoPtr> &params,
                            pdb::LogicalPlanPtr &plan) override {

    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(projection, plan, whereEveryoneGoes);
    return correctJoinTuple->getKeySink(consumeMe, attsToOpOn, projection, whereEveryoneGoes, numPartitions);
  }

  ComputeSinkPtr getComputeMerger(TupleSpec &consumeMe,
                                  TupleSpec &attsToOpOn,
                                  TupleSpec &projection,
                                  uint64_t workerID,
                                  uint64_t numThreads,
                                  uint64_t numNodes,
                                  pdb::LogicalPlanPtr &plan) override {

    // loop through each of the attributes that we are supposed to accept, and for each of them, find the type
    std::vector<std::string> typeList;
    AtomicComputationPtr producer = plan->getComputations().getProducingAtomicComputation(consumeMe.getSetName());

    for (auto &a : projection.getAtts()) {

      // find the identity of the producing computation
      std::pair<std::string, std::string> res = producer->findSource(a, plan->getComputations());

      // and find its type... in the first case, there is not a particular lambda that we need to ask for
      if (res.second.empty()) {
        typeList.push_back("pdb::Handle<" + plan->getNode(res.first).getComputation().getOutputType() + ">");
      } else {
        std::string myType = plan->getNode(res.first).getLambda(res.second)->getOutputType();
        if (myType.find_first_of("pdb::Handle<") == 0) {
          typeList.push_back(myType);
        } else {
          typeList.push_back("pdb::Handle<" + myType + ">");
        }
      }
    }

    // now we get the correct join tuple, that will allow us to pack tuples from the join in a hash table
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findCorrectJoinTuple<In1, In2, Rest...>(typeList, whereEveryoneGoes);

    // return the merger
    return correctJoinTuple->getBroadcastJoinHashMapCombiner(workerID, numThreads, numNodes);
  }

  PageProcessorPtr getShuffleJoinProcessor(size_t numNodes,
                                           size_t numProcessingThreads,
                                           vector<PDBPageQueuePtr> &pageQueues,
                                           PDBBufferManagerInterfacePtr &bufferManager,
                                           TupleSpec &recordSchema,
                                           pdb::LogicalPlanPtr &plan) override {

    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(recordSchema, plan, whereEveryoneGoes);

    // return the page processor
    return correctJoinTuple->getPageProcessor(numNodes, numProcessingThreads, pageQueues, bufferManager);
  }

  RHSShuffleJoinSourceBasePtr getRHSShuffleJoinSource(TupleSpec &inputSchema,
                                                      TupleSpec &hashSchema,
                                                      TupleSpec &recordSchema,
                                                      const PDBAbstractPageSetPtr &leftInputPageSet,
                                                      pdb::LogicalPlanPtr &plan,
                                                      uint64_t workerID) override {

    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(recordSchema, plan, whereEveryoneGoes);

    // return the lhs join source
    return correctJoinTuple->getRHSShuffleJoinSource(inputSchema,
                                                     hashSchema,
                                                     recordSchema,
                                                     leftInputPageSet,
                                                     whereEveryoneGoes,
                                                     workerID);
  }

  ComputeSourcePtr getJoinAggSource(int32_t nodeID,
                                    int32_t workerID,
                                    int32_t numWorkers,
                                    std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &leftTIDToRecordMapping,
                                    std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &rightTIDToRecordMapping,
                                    const PDBPageHandle &page,
                                    const JoinAggTupleEmitterPtr &emitter,
                                    PDBRandomAccessPageSetPtr leftInputPageSet,
                                    PDBRandomAccessPageSetPtr rightInputPageSet) override {

    if (sizeof...(Rest) == 0) {
      return std::make_shared<JoinAggSource<In1, In2>>(nodeID,
                                                       workerID,
                                                       numWorkers,
                                                       emitter,
                                                       leftInputPageSet,
                                                       rightInputPageSet);
    } else {
      throw runtime_error("");
    }
  }

  ComputeSourcePtr getTRAJoinSource(int32_t nodeID,
                                    int32_t workerID,
                                    int32_t numWorkers,
                                    std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &leftTIDToRecordMapping,
                                    std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> &rightTIDToRecordMapping,
                                    const PDBPageHandle &page,
                                    const JoinAggTupleEmitterPtr &emitter,
                                    PDBRandomAccessPageSetPtr leftInputPageSet,
                                    PDBRandomAccessPageSetPtr rightInputPageSet) override {

    if (sizeof...(Rest) == 0) {
      return std::make_shared<TRAJoinSource<In1, In2>>(nodeID,
                                                       workerID,
                                                       numWorkers,
                                                       emitter,
                                                       leftInputPageSet,
                                                       rightInputPageSet);
    } else {
      throw runtime_error("");
    }
  }

  ComputeSourcePtr getJoinedSource(TupleSpec &recordSchemaLHS,
                                   TupleSpec &inputSchemaRHS,
                                   TupleSpec &hashSchemaRHS,
                                   TupleSpec &recordSchemaRHS,
                                   RHSShuffleJoinSourceBasePtr leftSource,
                                   const PDBAbstractPageSetPtr &rightInputPageSet,
                                   pdb::LogicalPlanPtr &plan,
                                   bool needToSwapLHSAndRhs,
                                   uint64_t workerID) override {

    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(recordSchemaLHS, plan, whereEveryoneGoes);

    // return the lhs join source
    return correctJoinTuple->getJoinedSource(inputSchemaRHS,
                                             hashSchemaRHS,
                                             recordSchemaRHS,
                                             leftSource,
                                             rightInputPageSet,
                                             whereEveryoneGoes,
                                             needToSwapLHSAndRhs,
                                             workerID);
  }

  RHSKeyJoinSourceBasePtr getRHSKeyShuffleJoinSource(TupleSpec &inputSchema,
                                                     TupleSpec &hashSchema,
                                                     TupleSpec &recordSchema,
                                                     const PDBAbstractPageSetPtr &leftInputPageSet,
                                                     pdb::LogicalPlanPtr &plan,
                                                     const KeyJoinSourceArgsPtr &keySourceArgs) override {
    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(recordSchema, plan, whereEveryoneGoes);

    // return the lhs join source
    return correctJoinTuple->getRHSShuffleKeyJoinSource(inputSchema,
                                                        hashSchema,
                                                        recordSchema,
                                                        leftInputPageSet,
                                                        whereEveryoneGoes,
                                                        keySourceArgs);
  }

  ComputeSourcePtr getKeyedJoinedSource(TupleSpec &recordSchemaLHS,
                                        TupleSpec &inputSchemaRHS,
                                        TupleSpec &hashSchemaRHS,
                                        TupleSpec &recordSchemaRHS,
                                        RHSKeyJoinSourceBasePtr leftSource,
                                        const PDBAbstractPageSetPtr &rightInputPageSet,
                                        pdb::LogicalPlanPtr &plan,
                                        bool needToSwapLHSAndRhs,
                                        const KeyJoinSourceArgsPtr &keySourceArgs) override {
    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(recordSchemaLHS, plan, whereEveryoneGoes);

    // return the lhs join source
    return correctJoinTuple->getJoinedKeySource(inputSchemaRHS,
                                                hashSchemaRHS,
                                                recordSchemaRHS,
                                                leftSource,
                                                rightInputPageSet,
                                                whereEveryoneGoes,
                                                needToSwapLHSAndRhs,
                                                keySourceArgs);
  }

  JoinTuplePtr findJoinTuple(TupleSpec &recordSchema, LogicalPlanPtr &plan, vector<int> &whereEveryoneGoes) const {

    // get the producing atomic computation
    AtomicComputationPtr producer = plan->getComputations().getProducingAtomicComputation(recordSchema.getSetName());

    // figure out the types
    vector<string> typeList;
    for (auto &a : recordSchema.getAtts()) {

      // find the identity of the producing computation
      pair<string, string> res = producer->findSource(a, plan->getComputations());

      if (res.second.empty()) {
        typeList.push_back("pdb::Handle<" + plan->getNode(res.first).getComputation().getOutputType() + ">");
      } else {

        std::string myType = plan->getNode(res.first).getLambda(res.second)->getOutputType();
        if (myType.find_first_of("pdb::Handle<") == 0) {
          typeList.push_back(myType);
        } else {
          typeList.push_back("pdb::Handle<" + myType + ">");
        }
      }

      std::cout << "Type found : " << typeList.back() << std::endl;
    }

    //
    JoinTuplePtr correctJoinTuple = findCorrectJoinTuple<In1, In2, Rest...>(typeList, whereEveryoneGoes);
    return correctJoinTuple;
  }

  // this is a join computation
  std::string getComputationType() override {
    return std::string("JoinComp");
  }

  std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets, int computationLabel) override {

    // if a join is a source we generate a SCAN so that we can refer to this computation and use it's page set
    if (isSource) {
      return std::move(toSourceTCAP(computationLabel));
    }

    if (inputTupleSets.size() != getNumInputs()) {
      std::cout << "ERROR: inputTupleSet size is " << inputTupleSets.size() << " and not equivalent with Join's inputs "
                << getNumInputs() << std::endl;
      return "";
    }

    /**
     * 1. Generate the TCAP for the join predicate
     */

    // this is going to have info about the inputs
    MultiInputsBase multiInputsBase(this->getNumInputs());

    // update tuple set name for input sets
    for (unsigned int i = 0; i < inputTupleSets.size(); i++) {

      // set the name of the tuple set for the i-th position
      multiInputsBase.tupleSetNamesForInputs[i] = inputTupleSets[i].getTupleSetName();

      // set the columns for the i-th position
      multiInputsBase.inputColumnsForInputs[i] = inputTupleSets[i].getColumnNamesToKeep();

      // the the columns to apply for the i-th position
      multiInputsBase.inputColumnsToApplyForInputs[i] = inputTupleSets[i].getColumnNamesToApply();

      // setup all input names (the column name corresponding to input in tuple set)
      multiInputsBase.inputNames[i] = inputTupleSets[i].getColumnNamesToApply()[0];

      // we are keeping all the inputs
      multiInputsBase.inputColumnsToKeep.insert(multiInputsBase.inputNames[i]);
    }

    // we label the lambdas from zero
    int lambdaLabel = 0;

    // get the selection lambda
    std::string tcapString = "\n/* Apply join selection */\n";
    Lambda<bool> selectionLambda = callGetSelection<Derived, In1, In2, Rest...>(*static_cast<Derived *>(this));
    tcapString += selectionLambda.toTCAPString(lambdaLabel,
                                               "JoinComp",
                                               computationLabel,
                                               false,
                                               &multiInputsBase,
                                               true);



    /**
     * 2. Generate the TCAP for the join projection
     */

    // get the projection lambda and it's inputs
    tcapString += "\n/* Apply join projection*/\n";
    Lambda<Handle<Out>>
        projectionLambda = callGetProjection<Derived, Out, In1, In2, Rest...>(*static_cast<Derived *>(this));
    tcapString += projectionLambda.toTCAPString(lambdaLabel,
                                                "JoinComp",
                                                computationLabel,
                                                true,
                                                &multiInputsBase,
                                                false);

    //  get the output columns
    auto outputColumns = multiInputsBase.inputColumnsToApplyForInputs[0];
    assert(outputColumns.size() == 1);
    this->outputColumnToApply = outputColumns[0];

    // update the tuple set
    this->outputTupleSetName = multiInputsBase.tupleSetNamesForInputs[0];

    return tcapString;
  }

  // gets an execute that can run a scan join... needToSwapAtts is true if the atts that are currently stored in the hash table need to
  // come SECOND in the output tuple sets... hashedInputSchema tells us the schema for the attributes that are currently stored in the
  // hash table... pipelinedInputSchema tells us the schema for the attributes that will be coming through the pipeline...
  // pipelinedAttsToOperateOn is the identity of the hash attribute... pipelinedAttsToIncludeInOutput tells us the set of attributes
  // that are coming through the pipeline that we actually have to write to the output stream
  ComputeExecutorPtr getExecutor(bool needToSwapAtts,
                                 TupleSpec &hashedInputSchema,
                                 TupleSpec &pipelinedInputSchema,
                                 TupleSpec &pipelinedAttsToOperateOn,
                                 TupleSpec &pipelinedAttsToIncludeInOutput,
                                 JoinArgPtr &joinArg,
                                 uint64_t numNodes,
                                 uint64_t numProcessingThreads,
                                 uint64_t workerID,
                                 ComputePlan &computePlan) override {

    // loop through each of the attributes that we are supposed to accept, and for each of them, find the type
    std::vector<std::string> typeList;
    AtomicComputationPtr producer =
        computePlan.getPlan()->getComputations().getProducingAtomicComputation(hashedInputSchema.getSetName());
    for (auto &a : (hashedInputSchema.getAtts())) {

      // find the identity of the producing computation
      std::pair<std::string, std::string> res = producer->findSource(a, computePlan.getPlan()->getComputations());

      // and find its type... in the first case, there is not a particular lambda that we need to ask for
      if (res.second.empty()) {
        typeList.push_back(
            "pdb::Handle<" + computePlan.getPlan()->getNode(res.first).getComputation().getOutputType() + ">");
      } else {
        std::string myType = computePlan.getPlan()->getNode(res.first).getLambda(res.second)->getOutputType();
        if (myType.find_first_of("pdb::Handle<") == 0) {
          typeList.push_back(myType);
        } else {
          typeList.push_back("pdb::Handle<" + myType + ">");
        }
      }
    }

    // now we get the correct join tuple, that will allow us to pack tuples from the join in a hash table
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findCorrectJoinTuple<In1, In2, Rest...>(typeList, whereEveryoneGoes);

    // and return the correct probing code
    return correctJoinTuple->getProber(joinArg->hashTablePageSet,
                                       whereEveryoneGoes,
                                       pipelinedInputSchema,
                                       pipelinedAttsToOperateOn,
                                       pipelinedAttsToIncludeInOutput,
                                       numNodes,
                                       numProcessingThreads,
                                       workerID,
                                       needToSwapAtts);
  }

  ComputeExecutorPtr getExecutor(bool needToSwapAtts,
                                 TupleSpec &hashedInputSchema,
                                 TupleSpec &pipelinedInputSchema,
                                 TupleSpec &pipelinedAttsToOperateOn,
                                 TupleSpec &pipelinedAttsToIncludeInOutput) override {
    std::cout << "Currently, no pipelined version of the join doesn't take an arg.\n";
    exit(1);
  }
};

}

#endif
