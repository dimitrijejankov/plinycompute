//
// Created by dimitrije on 2/19/19.
//

#include "PDBComputationServerFrontend.h"
#include "HeapRequestHandler.h"
#include "CSExecuteComputation.h"

void pdb::PDBComputationServerFrontend::registerHandlers(pdb::PDBServer &forMe) {

  forMe.registerHandler(
      CSExecuteComputation_TYPEID,
      make_shared<pdb::HeapRequestHandler<pdb::CSExecuteComputation>>(
          [&](Handle<pdb::CSExecuteComputation> request, PDBCommunicatorPtr sendUsingMe) {

            for(int i = 0; i < request->computations->size(); i++) {
              std::cout << (*request->computations)[i]->getComputationType() << std::endl;
            }

            return make_pair(true, std::string(""));
          }));

}
