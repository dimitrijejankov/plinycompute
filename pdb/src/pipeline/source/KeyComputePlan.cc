#include <KeyComputePlan.h>

std::vector<AtomicComputationPtr> pdb::KeyComputePlan::getLeftPipelineComputations(AtomicComputationPtr &source,
                                                                                   shared_ptr<pdb::LogicalPlan> &logicalPlan) {
  // we are going to return this
  std::vector<AtomicComputationPtr> tmp;

  // get the current tuple set
  auto currTupleSet = source->getOutputName();

  // while we don't hit the hash computation loop
  while(!logicalPlan->getComputations().getConsumingAtomicComputations(currTupleSet).empty()) {

    // get all the consumers
    auto &consumers = logicalPlan->getComputations().getConsumingAtomicComputations(currTupleSet);

    // there has to be exactly one consumer otherwise something went wrong
    if(consumers.size() != 1) {
      return {};
    }

    // insert the consumer
    tmp.emplace_back(consumers.back());

    // check if this is a hash we are done
    auto compType = consumers.back()->getAtomicComputationTypeID();
    if( compType == HashLeftTypeID || compType == HashRightTypeID || compType == HashOneTypeID) {
      return std::move(tmp);
    }

    // update the tuple set for the next iteration
    currTupleSet = consumers.back()->getOutputName();
  }

  // we did not find a hash, return empty list
  return {};
}
//
//pdb::PipelinePtr pdb::KeyComputePlan::buildLeftPipeline(AtomicComputationPtr &source,
//                                                        const pdb::PDBAbstractPageSetPtr &inputPageSet,
//                                                        const pdb::PDBAnonymousPageSetPtr &outputPageSet,
//                                                        map<pdb::ComputeInfoType, pdb::ComputeInfoPtr> &params,
//                                                        shared_ptr<pdb::LogicalPlan> &myPlan) {
//
//  // get the left pipeline computations
//  auto listSoFar = getLeftPipelineComputations(source, myPlan);
//
//  // get the name of the producer set
//  auto producerName = source->getOutputName();
//
//  // our source is a normal source and not a join source, so we just grab it from the computation
//  auto computeSource = myPlan->getNode(producerName).getComputation().getComputeSource(inputPageSet, 1000, 0, params);
//
//  // returns the key sink if needed
//  ComputeSinkPtr computeSink = getSink(listSoFar.back(), params, myPlan);
//
//  // make the pipeline
//  std::shared_ptr<Pipeline> returnVal = std::make_shared<Pipeline>(outputPageSet, computeSource, computeSink, processor);
//
//  // make the pipeline
//  AtomicComputationPtr lastOne = myPlan->getComputations().getProducingAtomicComputation(source->getOutputName());
//  for (auto &a : listSoFar) {
//
//    // if we have a filter, then just go ahead and create it
//    if (a->getAtomicComputationType() == "Filter") {
//
//      // create a filter executor
//      std::cout << "Adding: " << a->getProjection() << " + filter [" << a->getInput() << "] => " << a->getOutput() << "\n";
//      returnVal->addStage(std::make_shared<FilterExecutor>(lastOne->getOutput(), a->getInput(), a->getProjection()));
//
//      // if we had an apply, go ahead and find it and add it to the pipeline
//    } else if (a->getAtomicComputationType() == "Apply") {
//
//      // create an executor for the apply lambda
//      std::cout << "Adding: " << a->getProjection() << " + apply [" << a->getInput() << "] => " << a->getOutput() << "\n";
//      returnVal->addStage(myPlan->getNode(a->getComputationName()).
//          getLambda(((ApplyLambda *) a.get())->getLambdaToApply())->getExecutor(lastOne->getOutput(), a->getInput(), a->getProjection()));
//
//    } else if(a->getAtomicComputationType() == "Union") {
//
//      // get the union
//      auto u = (Union *) a.get();
//
//      // check if we are pipelining the right input
//      if (lastOne->getOutput().getSetName() == u->getRightInput().getSetName()) {
//
//        std::cout << "Adding: " << " + apply [" << u->getInput() << ", " << u->getRightInput() << "] => " << u->getOutput() << "\n";
//        returnVal->addStage(std::make_shared<UnionExecutor>(lastOne->getOutput(), u->getRightInput()));
//
//      } else {
//
//        std::cout << "Adding: " << " + apply [" << u->getInput() << ", " << u->getInput() << "] => " << u->getOutput() << "\n";
//        returnVal->addStage(std::make_shared<UnionExecutor>(lastOne->getOutput(), u->getInput()));
//      }
//
//    } else if (a->getAtomicComputationType() == "HashLeft") {
//
//      // create an executor for left hasher
//      std::cout << "Adding: " << a->getProjection() << " + hashleft [" << a->getInput() << "] => " << a->getOutput() << "\n";
//      returnVal->addStage(myPlan->getNode(a->getComputationName()).
//          getLambda(((HashLeft *) a.get())->getLambdaToApply())->getLeftHasher(lastOne->getOutput(), a->getInput(), a->getProjection()));
//
//    } else if (a->getAtomicComputationType() == "HashRight") {
//
//      // create an executor for the right hasher
//      std::cout << "Adding: " << a->getProjection() << " + hashright [" << a->getInput() << "] => " << a->getOutput() << "\n";
//      returnVal->addStage(myPlan->getNode(a->getComputationName()).
//          getLambda(((HashLeft *) a.get())->getLambdaToApply())->getRightHasher(lastOne->getOutput(), a->getInput(), a->getProjection()));
//
//
//    } else if (a->getAtomicComputationType() == "HashOne") {
//
//      std::cout << "Adding: " << a->getProjection() << " + hashone [" << a->getInput() << "] => " << a->getOutput() << "\n";
//      returnVal->addStage(std::make_shared<HashOneExecutor>(lastOne->getOutput(), a->getInput(), a->getProjection()));
//
//    } else if (a->getAtomicComputationType() == "Flatten") {
//
//      std::cout << "Adding: " << a->getProjection() << " + flatten [" << a->getInput() << "] => " << a->getOutput() << "\n";
//      returnVal->addStage(std::make_shared<FlattenExecutor>(lastOne->getOutput(), a->getInput(), a->getProjection()));
//
//    } else if (a->getAtomicComputationType() == "JoinSets") {
//
//      std::cout << "Adding: " << a->getProjection() << " + join [" << a->getInput() << "] => " << a->getOutput() << "\n";
//
//      // join is weird, because there are two inputs...
//      auto &myComp = (JoinCompBase &) myPlan->getNode(a->getComputationName()).getComputation();
//      auto *myJoin = (ApplyJoin *) (a.get());
//
//      // grab the join arguments
//      JoinArgumentsPtr joinArgs = std::dynamic_pointer_cast<JoinArguments>(params[ComputeInfoType::JOIN_ARGS]);
//      if(joinArgs == nullptr) {
//        throw runtime_error("Join pipeline run without hash tables!");
//      }
//
//      // check if we are pipelining the right input
//      if (lastOne->getOutput().getSetName() == myJoin->getRightInput().getSetName()) {
//
//        // do we have the appropriate join arguments? if not throw an exception
//        auto it = joinArgs->hashTables.find(myJoin->getInput().getSetName());
//        if (it == joinArgs->hashTables.end()) {
//          throw runtime_error("Hash table for the output set," + a->getOutput().getSetName() + "not found!");
//        }
//
//        // if we are pipelining the right input, then we don't need to switch left and right inputs
//        std::cout << "We are pipelining the right input...\n";
//        returnVal->addStage(myComp.getExecutor(true, myJoin->getProjection(), lastOne->getOutput(), myJoin->getRightInput(), myJoin->getRightProjection(), it->second, 1, 1, 0, *this));
//      } else {
//        // do we have the appropriate join arguments? if not throw an exception
//        auto it = joinArgs->hashTables.find(myJoin->getRightInput().getSetName());
//        if (it == joinArgs->hashTables.end()) {
//          throw runtime_error("Hash table for the output set," + a->getOutput().getSetName() + "not found!");
//        }
//        // if we are pipelining the right input, then we don't need to switch left and right inputs
//        std::cout << "We are pipelining the left input...\n";
//        returnVal->addStage(myComp.getExecutor(false, myJoin->getRightProjection(), lastOne->getOutput(), myJoin->getInput(), myJoin->getProjection(), it->second, 1, 1, 0, *this));
//      }
//
//    }
//    else if(a->getAtomicComputationType() == "WriteSet") {
//
//      // skip this one
//      std::cout << "We are skipping a write set this is essentially a NOOP\n";
//    }
//    else {
//      std::cout << "This is bad... found an unexpected computation type (" << a->getAtomicComputationType()
//                << ") inside of a pipeline.\n";
//    }
//    lastOne = a;
//  }
//
//  return std::move(returnVal);
//}
