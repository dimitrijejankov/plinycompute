//
// Created by dimitrije on 2/19/19.
//

#include <boost/filesystem/path.hpp>
#include "PDBComputationServerFrontend.h"
#include "HeapRequestHandler.h"
#include "CSExecuteComputation.h"
#include "PDBPhysicalOptimizer.h"

void pdb::PDBComputationServerFrontend::init() {

  // init the class
  logger = make_shared<pdb::PDBLogger>((boost::filesystem::path(getConfiguration()->rootDirectory) / "logs").string(),
                                       "PDBComputationServerFrontend.log");
}

void pdb::PDBComputationServerFrontend::registerHandlers(pdb::PDBServer &forMe) {

  forMe.registerHandler(
      CSExecuteComputation_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::CSExecuteComputation>>(
          [&](Handle<pdb::CSExecuteComputation> request, PDBCommunicatorPtr sendUsingMe) {

            // the id associated with this computation
            auto compID = this->statsManager.startComputation();

            // init the optimizer
            pdb::PDBPhysicalOptimizer optimizer(request->tcapString, logger);

            // while we still have jobs to execute
            while(optimizer.hasAlgorithmToRun()) {

              // grab a algorithm
              auto algorithm = optimizer.getNextAlgorithm();

              /// TODO pack this into a job, broadcast the job and wait for it to finish...

              // update stats
              optimizer.updateStats();
            }

            // stop the computation
            this->statsManager.endComputation(compID);

            return make_pair(true, std::string(""));
          }));

}
