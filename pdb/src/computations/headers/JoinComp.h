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
#include "Computation.h"
#include "JoinTests.h"
#include "ComputePlan.h"
#include "JoinTuple.h"
#include "JoinCompBase.h"
#include "LogicalPlan.h"
#include "MultiInputsBase.h"
#include "Lambda.h"

namespace pdb {

template<typename Derived, typename Out, typename In1, typename In2, typename ...Rest>
class JoinComp : public JoinCompBase {
public:

  virtual ~JoinComp() {
    if (multiInputsBase == nullptr) {
      delete (multiInputsBase);
    }
  }

  // calls getProjection and getSelection to extract the lambdas
  void extractLambdas(std::map<std::string, LambdaObjectPtr> &returnVal) override {
    int suffix = 0;
    Lambda<bool> selectionLambda = callGetSelection<Derived, In1, In2, Rest...>(*static_cast<Derived*>(this));
    Lambda<Handle<Out>> projectionLambda = callGetProjection<Derived, In1, In2, Rest...>(*static_cast<Derived*>(this));
    selectionLambda.toMap(returnVal, suffix);
    projectionLambda.toMap(returnVal, suffix);
  }

  // return the output type
  std::string getOutputType() override {
    return getTypeName<Out>();
  }

  // count the number of inputs
  int getNumInputs() final {
    const int extras = sizeof...(Rest);
    return extras + 2;
  }

  MultiInputsBase* getMultiInputsBase() {
    if (multiInputsBase == nullptr) {
      multiInputsBase = new MultiInputsBase();
    }
    return multiInputsBase;
  }

  template<typename First, typename ...Others>
  typename std::enable_if<sizeof ...(Others) == 0, std::string>::type getIthInputType(int i) {
    if (i == 0) {
      return getTypeName<First>();
    } else {
      std::cout << "Asked for an input type that didn't exist!";
      exit(1);
    }
  }

  // helper function to get a particular intput type
  template<typename First, typename ...Others>
  typename std::enable_if<sizeof ...(Others) != 0, std::string>::type getIthInputType(int i) {
    if (i == 0) {
      return getTypeName<First>();
    } else {
      return getIthInputType<Others...>(i - 1);
    }
  }

  // from the interface: get the i^th input type
  std::string getIthInputType(int i) final {
    return getIthInputType<In1, In2, Rest...>(i);
  }

  // this gets a compute sink
  ComputeSinkPtr getComputeSink(TupleSpec &consumeMe,
                                TupleSpec &attsToOpOn,
                                TupleSpec &projection,
                                uint64_t numPartitions,
                                std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                pdb::LogicalPlanPtr &plan) override {

    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(projection, plan, whereEveryoneGoes);
    return correctJoinTuple->getSink(consumeMe, attsToOpOn, projection, whereEveryoneGoes, numPartitions);
  }

  ComputeSinkPtr getComputeMerger(TupleSpec &consumeMe, TupleSpec &attsToOpOn, TupleSpec &projection,
                                  uint64_t workerID, uint64_t numThreads, uint64_t numNodes, pdb::LogicalPlanPtr &plan) override {

    // loop through each of the attributes that we are supposed to accept, and for each of them, find the type
    std::vector<std::string> typeList;
    AtomicComputationPtr producer = plan->getComputations().getProducingAtomicComputation(consumeMe.getSetName());
    std::cout << "consumeMe was: " << consumeMe << "\n";
    std::cout << "attsToOpOn was: " << attsToOpOn << "\n";
    std::cout << "projection was: " << projection << "\n";
    for (auto &a : projection.getAtts()) {

      // find the identity of the producing computation
      std::cout << "finding the source of " << projection.getSetName() << "." << a << "\n";
      std::pair<std::string, std::string> res = producer->findSource(a, plan->getComputations());
      std::cout << "got " << res.first << " " << res.second << "\n";

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

    for (auto &aa : typeList) {
      std::cout << "Got type " << aa << "\n";
    }

    // now we get the correct join tuple, that will allow us to pack tuples from the join in a hash table
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findCorrectJoinTuple<In1, In2, Rest...>(typeList, whereEveryoneGoes);

    for (auto &aa : whereEveryoneGoes) {
      std::cout << aa << " ";
    }
    std::cout << "\n";

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
                                                      uint64_t chunkSize,
                                                      uint64_t workerID) override {

    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(recordSchema, plan, whereEveryoneGoes);

    // for debug
    for (auto &aa : whereEveryoneGoes) {
      std::cout << aa << " ";
    }
    std::cout << "\n";

    // return the lhs join source
    return correctJoinTuple->getRHSShuffleJoinSource(inputSchema,
                                                     hashSchema,
                                                     recordSchema,
                                                     leftInputPageSet,
                                                     whereEveryoneGoes,
                                                     chunkSize,
                                                     workerID);
  }

  ComputeSourcePtr getJoinedSource(TupleSpec &recordSchemaLHS,
                                   TupleSpec &inputSchemaRHS,
                                   TupleSpec &hashSchemaRHS,
                                   TupleSpec &recordSchemaRHS,
                                   RHSShuffleJoinSourceBasePtr leftSource,
                                   const PDBAbstractPageSetPtr &rightInputPageSet,
                                   pdb::LogicalPlanPtr &plan,
                                   bool needToSwapLHSAndRhs,
                                   uint64_t chunkSize,
                                   uint64_t workerID) override {

    // figure out the right join tuple
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findJoinTuple(recordSchemaLHS, plan, whereEveryoneGoes);

    // for debug
    for (auto &aa : whereEveryoneGoes) {
      std::cout << aa << " ";
    }
    std::cout << "\n";

    // return the lhs join source
    return correctJoinTuple->getJoinedSource(inputSchemaRHS, hashSchemaRHS, recordSchemaRHS, leftSource, rightInputPageSet, whereEveryoneGoes, needToSwapLHSAndRhs, chunkSize, workerID);
  }

  JoinTuplePtr findJoinTuple(TupleSpec &recordSchema, LogicalPlanPtr &plan, vector<int> &whereEveryoneGoes) const {

    // get the producing atomic computation
    AtomicComputationPtr producer = plan->getComputations().getProducingAtomicComputation(recordSchema.getSetName());

    // figure out the types
    vector<string> typeList;
    for (auto &a : recordSchema.getAtts()) {

      // find the identity of the producing computation
      cout << "finding the source of " << recordSchema.getSetName() << "." << a << "\n";
      pair<string, string> res = producer->findSource(a, plan->getComputations());
      cout << "got " << res.first << " " << res.second << "\n";

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

  //JiaNote: Returning a TCAP string for this Join computation
  std::string toTCAPString(std::vector<InputTupleSetSpecifier> inputTupleSets,
                           int computationLabel) override {
    if (inputTupleSets.size() == getNumInputs()) {
      std::string tcapString;
      if (multiInputsBase == nullptr) {
        multiInputsBase = new MultiInputsBase();
      }
      multiInputsBase->setNumInputs(this->getNumInputs());
      std::vector<std::string> inputNames;

      // update tupleset name for input sets
      for (unsigned int i = 0; i < inputTupleSets.size(); i++) {
        this->multiInputsBase->setTupleSetNameForIthInput(i, inputTupleSets[i].getTupleSetName());
        this->multiInputsBase->setInputColumnsForIthInput(i, inputTupleSets[i].getColumnNamesToKeep());
        this->multiInputsBase->setInputColumnsToApplyForIthInput(i, inputTupleSets[i].getColumnNamesToApply());
        inputNames.push_back(inputTupleSets[i].getColumnNamesToApply()[0]);
      }

      analyzeInputSets(inputNames);
      Lambda<bool> selectionLambda = callGetSelection<Derived, In1, In2, Rest...>(*static_cast<Derived*>(this));
      std::string inputTupleSetName;
      std::vector<std::string> inputColumnNames;
      std::vector<std::string> inputColumnsToApply;
      std::vector<std::string> childrenLambdaNames;
      int lambdaLabel = 0;
      std::string myLambdaName;

      // TODO
      std::string outputTupleSetName;
      std::vector<std::string> outputColumnNames;
      std::string addedOutputColumnName;

      MultiInputsBase *multiInputsComp = this->getMultiInputsBase();
      tcapString += selectionLambda.toTCAPString(inputTupleSetName,
                                                 inputColumnNames,
                                                 inputColumnsToApply,
                                                 childrenLambdaNames,
                                                 lambdaLabel,
                                                 "JoinComp",
                                                 computationLabel,
                                                 outputTupleSetName,
                                                 outputColumnNames,
                                                 addedOutputColumnName,
                                                 myLambdaName,
                                                 false,
                                                 multiInputsComp,
                                                 true);

      std::vector<std::string> inputsInProjection = multiInputsComp->getInputsInProjection();
      tcapString += "\n/* run Join projection on ( " + inputsInProjection[0];
      for (unsigned int i = 1; i < inputsInProjection.size(); i++) {
        tcapString += " " + inputsInProjection[i];
      }
      tcapString += " )*/\n";
      Lambda<Handle<Out>> projectionLambda = callGetProjection<Derived, In1, In2, Rest...>(*static_cast<Derived*>(this));
      inputTupleSetName = outputTupleSetName;
      inputColumnNames.clear();
      inputColumnsToApply.clear();
      childrenLambdaNames.clear();
      for (unsigned int index = 0; index < multiInputsComp->getNumInputs(); index++) {
        multiInputsComp->setInputColumnsForIthInput(index, inputColumnNames);
      }

      tcapString += projectionLambda.toTCAPString(inputTupleSetName,
                                                  inputColumnNames,
                                                  inputColumnsToApply,
                                                  childrenLambdaNames,
                                                  lambdaLabel,
                                                  "JoinComp",
                                                  computationLabel,
                                                  outputTupleSetName,
                                                  outputColumnNames,
                                                  addedOutputColumnName,
                                                  myLambdaName,
                                                  true,
                                                  multiInputsComp,
                                                  false);

      this->outputTupleSetName = outputTupleSetName;
      this->outputColumnToApply = addedOutputColumnName;
      setMultiInputsBaseToNull();
      return tcapString;

    } else {
      std::cout << "ERROR: inputTupleSet size is " << inputTupleSets.size()
                << " and not equivalent with Join's inputs " << getNumInputs() << std::endl;
      return "";
    }
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

    std::cout << "pipelinedInputSchema is " << pipelinedInputSchema << "\n";
    std::cout << "pipelinedAttsToOperateOn is " << pipelinedAttsToOperateOn << "\n";
    std::cout << "pipelinedAttsToIncludeInOutput is " << pipelinedAttsToIncludeInOutput << "\n";
    //std::cout << "From the join arg, got " << joinArg->hashTablePageSet->getNumPages() << "\n";

    // loop through each of the attributes that we are supposed to accept, and for each of them, find the type
    std::vector<std::string> typeList;
    AtomicComputationPtr producer =
        computePlan.getPlan()->getComputations().getProducingAtomicComputation(hashedInputSchema.getSetName());
    for (auto &a : (hashedInputSchema.getAtts())) {

      // find the identity of the producing computation
      std::cout << "finding the source of " << hashedInputSchema.getSetName() << "." << a << "\n";
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

    for (auto &aa : typeList) {
      std::cout << "Got type " << aa << "\n";
    }

    // now we get the correct join tuple, that will allow us to pack tuples from the join in a hash table
    std::vector<int> whereEveryoneGoes;
    JoinTuplePtr correctJoinTuple = findCorrectJoinTuple<In1, In2, Rest...>(typeList, whereEveryoneGoes);

    std::cout << "whereEveryoneGoes was: ";
    for (auto &a : whereEveryoneGoes) {
      std::cout << a << " ";
    }
    std::cout << "\n";

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

  void analyzeInputSets(std::vector<std::string> &inputNames) {
    if (multiInputsBase == nullptr) {
      multiInputsBase = new MultiInputsBase();
    }
    // Step 1. setup all input names (the column name corresponding to input in tuple set)
    for (int i = 0; i < inputNames.size(); i++) {
      this->multiInputsBase->setNameForIthInput(i, inputNames[i]);
    }

    // Step 2. analyze selectionLambda to find all inputs in predicates
    Lambda<bool> selectionLambda = callGetSelection<Derived, In1, In2, Rest...>(*static_cast<Derived*>(this));
    std::vector<std::string> inputsInPredicates =
        selectionLambda.getAllInputs(this->multiInputsBase);
     for (auto &inputsInPredicate : inputsInPredicates) {
      this->multiInputsBase->addInputNameToPredicates(inputsInPredicate);
    }
    // Step 3. analyze projectionLambda to find all inputs in projection
    Lambda<Handle<Out>> projectionLambda = callGetProjection<Derived, In1, In2, Rest...>(*static_cast<Derived*>(this));
    std::vector<std::string> inputsInProjection =
        projectionLambda.getAllInputs(this->multiInputsBase);
    for (auto &i : inputsInProjection) {
      this->multiInputsBase->addInputNameToProjection(i);
    }
  }

  void setMultiInputsBaseToNull() {
    if (multiInputsBase == nullptr) {
      delete (multiInputsBase);
    }
    multiInputsBase = nullptr;
  }

private:

  MultiInputsBase *multiInputsBase = nullptr;
};

}

#endif
