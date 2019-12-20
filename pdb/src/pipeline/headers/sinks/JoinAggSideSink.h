#pragma once

#include <ComputeSink.h>
#include <TupleSpec.h>
#include <TupleSetMachine.h>
#include <JoinMap.h>
#include <JoinTuple.h>
#include <PDBCommunicator.h>

#include <utility>
#include <PipJoinAggPlanResult.h>
#include "../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlockMeta.h"
#include "../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlock.h"
#include "../../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlockMeta.h"

namespace pdb {

// this class is used to create a ComputeSink object that stores special objects that wrap up multiple columns of a tuple
template<typename RHSType>
class JoinAggSideSink : public ComputeSink {

private:

  // tells us which attribute is the key which one the value
  int keyAtt;
  int valAtt;

  // if useTheseAtts[i] = j, it means that the i^th attribute that we need to extract from the input tuple is j
  std::vector<int> useTheseAtts;

  // if whereEveryoneGoes[i] = j, it means that the i^th entry in useTheseAtts goes in the j^th pos in the holder tuple
  std::vector<int> whereEveryoneGoes;

  // this is the list of columns that we are processing
  void **columns = nullptr;

  // the number of partitions
  size_t numPartitions;

  // the senders
  std::shared_ptr<std::vector<JoinAggSideSenderPtr>> sendersToNode;

  // the id we get for waiting
  std::vector<int32_t> sendingRequestID;

  // the left to key page
  pdb::PDBPageHandle keyToLeft;

  // the plan page
  pdb::PDBPageHandle planPage;

  // is it the left side or right side
  bool isLeft;

  // is this initialized or not?
  bool isInitialized = false;

  // basically stores
  std::vector<std::vector<std::pair<uint32_t, pdb::Handle<pdb::matrix::MatrixBlock>>>> nodeToRecord;

  //
  Handle<pdb::Map<pdb::matrix::MatrixBlockMeta, uint32_t>> tidMap;

  // the plan of for the
  Handle<PipJoinAggPlanResult::JoinTIDToNode> plan;

  // the buffer page
  PDBPageHandle bufferPage;

public:

  JoinAggSideSink(TupleSpec &inputSchema,
                  TupleSpec &attsToOperateOn,
                  TupleSpec &additionalAtts,
                  std::vector<int> &whereEveryoneGoes,
                  pdb::PDBPageHandle keyToLeft,
                  std::shared_ptr<std::vector<JoinAggSideSenderPtr>> senders,
                  pdb::PDBPageHandle planPage,
                  bool isLeft,
                  size_t numPartitions) : numPartitions(numPartitions),
                                          whereEveryoneGoes(whereEveryoneGoes),
                                          keyToLeft(std::move(keyToLeft)),
                                          sendersToNode(std::move(senders)),
                                          planPage(std::move(planPage)),
                                          isLeft(isLeft) {

    // used to manage attributes and set up the output
    TupleSetSetupMachine myMachine(inputSchema);

    // figure out the key att
    std::vector<int> matches = myMachine.match(inputSchema);
    valAtt = matches[0];
    keyAtt = matches[1];
  }

  ~JoinAggSideSink() override {
    if (columns != nullptr)
      delete[] columns;
  }

  Handle<Object> createNewOutputContainer() override {
    return nullptr;
  }

  void writeOut(TupleSetPtr input, Handle<Object> &writeToMe) override {

    std::vector<pdb::Handle<pdb::matrix::MatrixBlock>> &valueColumns = input->getColumn<pdb::Handle<pdb::matrix::MatrixBlock>>(valAtt);
    std::vector<pdb::Handle<pdb::matrix::MatrixBlockMeta>> &keyColumns = input->getColumn<pdb::Handle<pdb::matrix::MatrixBlockMeta>>(keyAtt);

    if(!this->isInitialized) {

      // repin the page for the tuple key mappings
      this->keyToLeft->repin();

      // grab the tid map from tid
      auto* tidMapRecord = (Record<pdb::Map<pdb::matrix::MatrixBlockMeta, uint32_t>>*) this->keyToLeft->getBytes();
      tidMap = tidMapRecord->getRootObject();

      // repin the plan page
      this->planPage->repin();

      // get the
      auto* recordCopy = (Record<PipJoinAggPlanResult>*) this->planPage->getBytes();

      // depending on the side select the right mapping
      if(isLeft) {
        this->plan = recordCopy->getRootObject()->leftToNode;
      }
      else {
        this->plan = recordCopy->getRootObject()->rightToNode;
      }

      // init the node to record
      nodeToRecord.resize(sendersToNode->size());
      sendingRequestID.resize(sendersToNode->size());
    }

    // go through all the value
    for(int i = 0; i < valueColumns.size(); ++i) {

      // get the right key
      auto &k = keyColumns[i];

      // get the right value
      auto &v = valueColumns[i];

      // get the tid
      auto tid = (*tidMap)[*k];
      std::cout << "The tid for " << k->

      rowID << ", " << k->colID << "->" << tid << " is sent to : ";

      // get the nodes vector
      Vector<bool> &nodes = (*plan)[tid];
      for(int j = 0; j < nodes.size(); ++j) {
        if(nodes[j]) {
          nodeToRecord[j].emplace_back(std::pair{tid, v});
          std::cout << j << " ";
        }
      }
      std::cout << "\n";
    }

    // queue them for sending
    for(int i = 0; i < sendersToNode->size(); ++i) {
      sendingRequestID[i] = (*sendersToNode)[i]->queueToSend(&nodeToRecord[i]);
    }

    // wait for all the senders to finish
    for(int i = 0; i < sendersToNode->size(); ++i) {

      // wait for the senders to finish sending our stuff
      (*sendersToNode)[i]->waitToFinish(sendingRequestID[i]);

      // clear the nodes
      nodeToRecord[i].clear();
    }
  }

  void writeOutPage(pdb::PDBPageHandle &page, Handle<Object> &writeToMe) override { throw runtime_error("JoinAggSink sink can not write out a page."); }

  // returns the number of records in the join sink
  uint64_t getNumRecords(Handle<Object> &writeToMe) override {
    // return the size
    return 1;
  }
};

}