#pragma once

#include <Join8SideReader.h>
#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "Join3KeyPipeline.h"
#include "Join8SideSender.h"
#include "Join8MapCreator.h"
#include "Pipeline.h"

namespace pdb {

struct PDBJoin3AlgorithmState : public PDBPhysicalAlgorithmState {

  // the page sets we get the keys from
  std::unordered_map<int32_t , PDBAbstractPageSetPtr> keySourcePageSets0;
  std::unordered_map<int32_t , PDBAbstractPageSetPtr> keySourcePageSets1;
  std::unordered_map<int32_t , PDBAbstractPageSetPtr> keySourcePageSets2;

  std::shared_ptr<Join3KeyPipeline> keyPipeline;

  //
  std::shared_ptr<std::vector<PDBPageQueuePtr>> planPageQueues = nullptr;

  std::shared_ptr<std::vector<PDBPageNetworkSenderPtr>> planSenders;

  std::shared_ptr<std::vector<PDBCommunicatorPtr>> joinSideCommunicatorsOut = nullptr;

  std::shared_ptr<std::vector<PDBCommunicatorPtr>> joinSideCommunicatorsIn = nullptr;

  std::shared_ptr<std::vector<Join8SideSenderPtr>> joinSideSenders = nullptr;

  std::shared_ptr<std::vector<Join8SideReaderPtr>> joinSideReader = nullptr;

  std::vector<std::multimap<uint32_t, std::tuple<uint32_t, uint32_t>>> TIDToRecordMapping;

  pdb::PDBFeedingPageSetPtr planPageSet = nullptr;

  std::shared_ptr<std::vector<Join8MapCreatorPtr>> joinMapCreators = nullptr;

  pdb::PDBRandomAccessPageSetPtr shuffledPageSet = nullptr;

  PDBPageHandle planPage;

  std::shared_ptr<std::vector<PipelinePtr>> myPipelines = nullptr;

  std::vector<Join3KeyPipeline::joined_record> joinedRecords;

  std::shared_ptr<std::vector<PDBCommunicatorPtr>> aJoinSideCommunicatorsOut = nullptr;

  std::shared_ptr<std::vector<PDBCommunicatorPtr>> aJoinSideCommunicatorsIn = nullptr;

  std::shared_ptr<std::vector<PDBCommunicatorPtr>> bJoinSideCommunicatorsOut = nullptr;

  std::shared_ptr<std::vector<PDBCommunicatorPtr>> bJoinSideCommunicatorsIn = nullptr;

  std::shared_ptr<std::vector<PDBCommunicatorPtr>> cJoinSideCommunicatorsOut = nullptr;

  std::shared_ptr<std::vector<PDBCommunicatorPtr>> cJoinSideCommunicatorsIn = nullptr;

};

}