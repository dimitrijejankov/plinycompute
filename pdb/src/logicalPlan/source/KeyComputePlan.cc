#include <utility>

#include <KeyComputePlan.h>
#include <ComputePlan.h>
#include <processors/NullProcessor.h>
#include <AtomicComputationClasses.h>
#include <JoinArguments.h>
#include <JoinCompBase.h>
#include <AggregateCompBase.h>

pdb::KeyComputePlan::KeyComputePlan(pdb::LogicalPlanPtr myPlan) : ComputePlan(std::move(myPlan)) {}

std::vector<AtomicComputationPtr> pdb::KeyComputePlan::getLeftPipelineComputations(AtomicComputationPtr &source) {

  // we are going to return this
  std::vector<AtomicComputationPtr> tmp;

  // get the current tuple set
  auto currTupleSet = source->getOutputName();

  // while we don't hit the join computation loop
  while(!myPlan->getComputations().getConsumingAtomicComputations(currTupleSet).empty()) {

    // get all the consumers
    auto &consumers = myPlan->getComputations().getConsumingAtomicComputations(currTupleSet);

    // there has to be exactly one consumer otherwise something went wrong
    if(consumers.size() != 1 ) {
      return {};
    }

    // insert the consumer
    tmp.emplace_back(consumers.back());

    // check if this is a join we are done
    auto compType = consumers.back()->getAtomicComputationTypeID();
    if(compType == HashOneTypeID || compType == HashLeftTypeID || compType == HashRightTypeID) {
      return std::move(tmp);
    }

    // update the tuple set for the next iteration
    currTupleSet = consumers.back()->getOutputName();
  }

  // we did not find a hash, return empty list
  return {};
}

pdb::PipelinePtr pdb::KeyComputePlan::buildHashPipeline(const std::string &sourceTupleSet,
                                                        const pdb::PDBAbstractPageSetPtr &inputPageSet,
                                                        const pdb::PDBAbstractPageSetPtr &outputPageSet,
                                                        map<pdb::ComputeInfoType, pdb::ComputeInfoPtr> &params) {

  //
  auto source = myPlan->getComputations().getProducingAtomicComputation(sourceTupleSet);

  // get the left pipeline computations
  auto listSoFar = getLeftPipelineComputations(source);

  // if there is nothing in the pipeline finish
  if(listSoFar.empty()) {
    return nullptr;
  }

  /// 1. Figure out the source

  // our source is a normal source and not a join source, so we just grab it from the computation
  auto computeSource = myPlan->getNode(source->getComputationName()).getComputation().getComputeSource(inputPageSet, 0, params);

  /// 2. Figure out the sink

  // get the atomic computation of the hasher
  auto &targetAtomicComp = listSoFar.back();

  // get the computation that corresponds to the hasher
  auto targetComputationName = targetAtomicComp->getComputationName();

  // returns the input specifier
  auto specifier = getSinkSpecifier(targetAtomicComp, targetComputationName);

  // returns the key sink
  auto computeSink = myPlan->getNode(targetComputationName).getComputation().getKeySink(std::get<0>(specifier),
                                                                                        std::get<1>(specifier),
                                                                                        std::get<2>(specifier),
                                                                                        1,
                                                                                        params,
                                                                                        myPlan);

  /// 3. Assemble the pipeline

  // assemble the whole pipeline
  return assemblePipeline(source->getOutputName(),
                          outputPageSet,
                          computeSource,
                          computeSink,
                          std::make_shared<NullProcessor>(),
                          params,
                          listSoFar,
                          1,
                          1,
                          0);
}

pdb::PipelinePtr pdb::KeyComputePlan::buildJoinAggPipeline(const std::string& sourceTupleSetName,
                                                           const std::string& targetTupleSetName,
                                                           const PDBAbstractPageSetPtr &inputPageSet,
                                                           const PDBAnonymousPageSetPtr &outputPageSet,
                                                           const PDBPageHandle &aggKeyPage,
                                                           std::map<ComputeInfoType, ComputeInfoPtr> &params,
                                                           size_t numNodes,
                                                           size_t numProcessingThreads,
                                                           uint64_t chunkSize,
                                                           uint64_t workerID) {

  // get all of the computations
  AtomicComputationList &allComps = myPlan->getComputations();

  /// 0. Figure out the compute source

  // get the atomic computation of the source
  auto sourceAtomicComputation = allComps.getProducingAtomicComputation(sourceTupleSetName);

  // and get the schema for the output TupleSet objects that it is supposed to produce
  TupleSpec &origSpec = sourceAtomicComputation->getOutput();

  // figure out the source
  ComputeSourcePtr computeSource = getKeySource(sourceAtomicComputation, inputPageSet, params);

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

  // get the compute sink
  auto computeSink = getJoinAggSink(targetAtomicComp,
                                    aggKeyPage,
                                    targetComputationName,
                                    params);

  // do we have a processor provided
  PageProcessorPtr processor = dynamic_pointer_cast<PageProcessor>(params.find(ComputeInfoType::PAGE_PROCESSOR)->second);

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

pdb::ComputeSinkPtr pdb::KeyComputePlan::getJoinAggSink(AtomicComputationPtr &targetAtomicComp,
                                                        const PDBPageHandle &aggKeyPage,
                                                        const std::string &targetComputationName,
                                                        std::map<ComputeInfoType, ComputeInfoPtr> &params) {

  // get the computation
  auto &comp = myPlan->getNode(targetComputationName).getComputation();
  if(comp.getComputationType() != "AggregationComp") {
    throw runtime_error("The aggregation in the join aggregation pipeline is not an aggregation.");
  }

  // now we have the list of computations, and so it is time to build the pipeline... start by building a compute sink
  return ((AggregateCompBase*) &comp)->getKeyJoinAggSink(targetAtomicComp->getOutput(),
                                                         targetAtomicComp->getInput(),
                                                         targetAtomicComp->getProjection(),
                                                         aggKeyPage,
                                                         params,
                                                         myPlan);
}

pdb::ComputeSourcePtr pdb::KeyComputePlan::getKeySource(AtomicComputationPtr &sourceAtomicComputation,
                                                        const PDBAbstractPageSetPtr &inputPageSet,
                                                        std::map<ComputeInfoType, ComputeInfoPtr> &params) {


  // now we get the name of the actual computation object that corresponds to the producer of this tuple set
  std::string producerName = sourceAtomicComputation->getComputationName();

  // get a reference to the computations of the logical plan
  auto &allComps = myPlan->getComputations();

  // if we are a join (shuffle join source) we need to have separate logic to handle that, otherwise just return a regular source
  if(sourceAtomicComputation->getAtomicComputationTypeID() != ApplyJoinTypeID) {
    throw runtime_error("Join Aggregation pipeline run without a join as a source!");
  }

  // cast the join computation
  auto *joinComputation = (ApplyJoin *) sourceAtomicComputation.get();

  // get the join source
  KeyJoinSourceArgsPtr keySourceArgs = std::dynamic_pointer_cast<KeyJoinSourceArgs>(params[ComputeInfoType::KEY_JOIN_SOURCE_ARGS]);

  // grab the join arguments
  JoinArgumentsPtr joinArgs = std::dynamic_pointer_cast<JoinArguments>(params[ComputeInfoType::JOIN_ARGS]);
  if(joinArgs == nullptr) {
    throw runtime_error("Join pipeline run without hash tables!");
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
    auto rhsSource = ((JoinCompBase *) &myPlan->getNode(joinComputation->getComputationName()).getComputation())->getRHSKeyShuffleJoinSource(rightAtomicComp->getOutput(),
                                                                                                                                             joinComputation->getRightInput(),
                                                                                                                                             joinComputation->getRightProjection(),
                                                                                                                                             it->second->hashTablePageSet,
                                                                                                                                             myPlan,
                                                                                                                                             keySourceArgs);

    // init the compute source for the join
    return ((JoinCompBase *) &myPlan->getNode(joinComputation->getComputationName()).getComputation())->getKeyedJoinedSource(joinComputation->getProjection(), // this tells me how the join tuple of the LHS is layed out
                                                                                                                             rightAtomicComp->getOutput(), // this gives the specification of the RHS tuple
                                                                                                                             joinComputation->getRightInput(), // this gives the location of the RHS hash
                                                                                                                             joinComputation->getRightProjection(), // this gives the projection of the RHS tuple
                                                                                                                             rhsSource, // the RHS source that gives us the tuples
                                                                                                                             inputPageSet, // the LHS page set
                                                                                                                             myPlan,
                                                                                                                             needsToSwapSides,
                                                                                                                             keySourceArgs);

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
    auto rhsSource = ((JoinCompBase *) &myPlan->getNode(joinComputation->getComputationName()).getComputation())->getRHSKeyShuffleJoinSource(rightAtomicComp->getOutput(),
                                                                                                                                             joinComputation->getInput(),
                                                                                                                                             joinComputation->getProjection(),
                                                                                                                                             it->second->hashTablePageSet,
                                                                                                                                             myPlan,
                                                                                                                                             keySourceArgs);

    // init the compute source for the join
    return ((JoinCompBase *) &myPlan->getNode(joinComputation->getComputationName()).getComputation())->getKeyedJoinedSource(joinComputation->getRightProjection(), // this tells me how the join tuple of the LHS is layed out
                                                                                                                             rightAtomicComp->getOutput(), // this gives the specification of the RHS tuple
                                                                                                                             joinComputation->getInput(), // this gives the location of the RHS hash
                                                                                                                             joinComputation->getProjection(), // this gives the projection of the RHS tuple
                                                                                                                             rhsSource, // the RHS source that gives us the tuples
                                                                                                                             inputPageSet, // the LHS page set
                                                                                                                             myPlan,
                                                                                                                             needsToSwapSides,
                                                                                                                             keySourceArgs);
  }
}
