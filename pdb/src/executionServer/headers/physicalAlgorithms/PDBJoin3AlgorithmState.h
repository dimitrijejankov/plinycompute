#pragma once

#include <Join8SideReader.h>
#include "PDBPhysicalAlgorithmState.h"
#include "PDBPageSelfReceiver.h"
#include "PDBPageNetworkSender.h"
#include "EightWayJoinPipeline.h"
#include "Join8SideSender.h"
#include "Join8MapCreator.h"
#include "Pipeline.h"

namespace pdb {

struct PDBJoin3AlgorithmState : public PDBPhysicalAlgorithmState {

  // the page sets we get the keys from
  std::unordered_map<int32_t , PDBAbstractPageSetPtr> keySourcePageSets;

  std::shared_ptr<EightWayJoinPipeline> keyPipeline;

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

  std::vector<EightWayJoinPipeline::joined_record> joinedRecords;

};

}