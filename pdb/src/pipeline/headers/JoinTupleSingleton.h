//
// Created by dimitrije on 2/19/19.
//

#ifndef PDB_JOINTUPLESINGLETON_H
#define PDB_JOINTUPLESINGLETON_H

#include <TupleSpec.h>
#include <ComputeExecutor.h>
#include <ComputeSink.h>
#include <JoinProbeExecutor.h>
#include <JoinSink.h>
#include <RHSShuffleJoinSource.h>
#include <JoinedShuffleJoinSource.h>
#include <processors/ShuffleJoinProcessor.h>
#include <BroadcastJoinCombinerSink.h>

namespace pdb {

// all join singletons descend from this
class JoinTupleSingleton {

 public:

  virtual ComputeExecutorPtr getProber(PDBAbstractPageSetPtr &hashTable,
                                       std::vector<int> &positions,
                                       TupleSpec &inputSchema,
                                       TupleSpec &attsToOperateOn,
                                       TupleSpec &attsToIncludeInOutput,
                                       uint64_t numNodes,
                                       uint64_t numProcessingThreads,
                                       uint64_t workerID,
                                       bool needToSwapLHSAndRhs) = 0;

  virtual ComputeSinkPtr getSink(TupleSpec &consumeMe,
                                 TupleSpec &attsToOpOn,
                                 TupleSpec &projection,
                                 std::vector<int> whereEveryoneGoes,
                                 uint64_t numPartitions) = 0;


  virtual RHSShuffleJoinSourceBasePtr getRHSShuffleJoinSource(TupleSpec &inputSchema,
                                                              TupleSpec &hashSchema,
                                                              TupleSpec &recordSchema,
                                                              const PDBAbstractPageSetPtr &leftInputPageSet,
                                                              std::vector<int> &recordOrder,
                                                              uint64_t chunkSize,
                                                              uint64_t workerID) = 0;

  virtual ComputeSourcePtr getJoinedSource(TupleSpec &inputSchemaRHS,
                                           TupleSpec &hashSchemaRHS,
                                           TupleSpec &recordSchemaRHS,
                                           RHSShuffleJoinSourceBasePtr rhsSource,
                                           const PDBAbstractPageSetPtr &lhsInputPageSet,
                                           std::vector<int> &lhsRecordOrder,
                                           bool needToSwapLHSAndRhs,
                                           uint64_t chunkSize,
                                           uint64_t workerID) = 0;

  virtual PageProcessorPtr getPageProcessor(size_t numNodes,
                                            size_t numProcessingThreads,
                                            vector<PDBPageQueuePtr> &pageQueues,
                                            PDBBufferManagerInterfacePtr &bufferManager) = 0;

  virtual ComputeSinkPtr getBroadcastJoinHashMapCombiner(uint64_t workerID, uint64_t numThreads, uint64_t numNodes) = 0;
};

// this is an actual class
template<typename HoldMe>
class JoinSingleton : public JoinTupleSingleton {

  // the actual data that we hold
  HoldMe myData;

 public:

  // gets a hash table prober

  ComputeExecutorPtr getProber(PDBAbstractPageSetPtr &hashTable,
                               std::vector<int> &positions,
                               TupleSpec &inputSchema,
                               TupleSpec &attsToOperateOn,
                               TupleSpec &attsToIncludeInOutput,
                               uint64_t numNodes,
                               uint64_t numProcessingThreads,
                               uint64_t workerID,
                               bool needToSwapLHSAndRhs) override {

    return std::make_shared<JoinProbeExecution<HoldMe>>(hashTable,
                                                        positions,
                                                        inputSchema,
                                                        attsToOperateOn,
                                                        attsToIncludeInOutput,
                                                        numNodes,
                                                        numProcessingThreads,
                                                        workerID,
                                                        needToSwapLHSAndRhs);
  }

  // creates a compute sink for this particular type
  ComputeSinkPtr getSink(TupleSpec &consumeMe,
                         TupleSpec &attsToOpOn,
                         TupleSpec &projection,
                         std::vector<int> whereEveryoneGoes,
                         uint64_t numPartitions) override {
    return std::make_shared<JoinSink<HoldMe>>(consumeMe, attsToOpOn, projection, whereEveryoneGoes, numPartitions);
  }


  RHSShuffleJoinSourceBasePtr getRHSShuffleJoinSource(TupleSpec &inputSchema,
                                                      TupleSpec &hashSchema,
                                                      TupleSpec &recordSchema,
                                                      const PDBAbstractPageSetPtr &leftInputPageSet,
                                                      std::vector<int> &recordOrder,
                                                      uint64_t chunkSize,
                                                      uint64_t workerID) override {

    return std::make_shared<RHSShuffleJoinSource<HoldMe>>(inputSchema, hashSchema, recordSchema, recordOrder, leftInputPageSet, chunkSize, workerID);
  }

  ComputeSourcePtr getJoinedSource(TupleSpec &inputSchemaRHS,
                                   TupleSpec &hashSchemaRHS,
                                   TupleSpec &recordSchemaRHS,
                                   RHSShuffleJoinSourceBasePtr rhsSource,
                                   const PDBAbstractPageSetPtr &lhsInputPageSet,
                                   std::vector<int> &lhsRecordOrder,
                                   bool needToSwapLHSAndRhs,
                                   uint64_t chunkSize,
                                   uint64_t workerID) override {

    /// remove this
    return std::make_shared<JoinedShuffleJoinSource<HoldMe>>(inputSchemaRHS, hashSchemaRHS, recordSchemaRHS, lhsInputPageSet, lhsRecordOrder, rhsSource, needToSwapLHSAndRhs, chunkSize, workerID);
  }

  PageProcessorPtr getPageProcessor(size_t numNodes,
                                    size_t numProcessingThreads,
                                    vector<PDBPageQueuePtr> &pageQueues,
                                    PDBBufferManagerInterfacePtr &bufferManager) override {
    return std::make_shared<ShuffleJoinProcessor<HoldMe>>(numNodes, numProcessingThreads, pageQueues, bufferManager);
  }

  ComputeSinkPtr getBroadcastJoinHashMapCombiner(uint64_t workerID, uint64_t numThreads, uint64_t numNodes) override {
    return std::make_shared<BroadcastJoinCombinerSink<HoldMe>>(workerID, numThreads, numNodes);
  }
};

typedef std::shared_ptr<JoinTupleSingleton> JoinTuplePtr;

inline int findType(std::string &findMe, std::vector<std::string> &typeList) {

  for (int i = 0; i < typeList.size(); i++) {
    if (typeList[i] == findMe) {
      typeList[i] = std::string("");
      return i;
    }
  }
  return -1;
}

template<typename In1>
typename std::enable_if<std::is_base_of<JoinTupleBase, In1>::value, JoinTuplePtr>::type findCorrectJoinTuple
    (std::vector<std::string> &typeList, std::vector<int> &whereEveryoneGoes);

template<typename In1, typename ...Rest>
typename std::enable_if<sizeof ...(Rest) != 0 && !std::is_base_of<JoinTupleBase, In1>::value,
                        JoinTuplePtr>::type findCorrectJoinTuple
    (std::vector<std::string> &typeList, std::vector<int> &whereEveryoneGoes);

template<typename In1, typename In2, typename ...Rest>
typename std::enable_if<std::is_base_of<JoinTupleBase, In1>::value, JoinTuplePtr>::type findCorrectJoinTuple
    (std::vector<std::string> &typeList, std::vector<int> &whereEveryoneGoes);

template<typename In1>
typename std::enable_if<!std::is_base_of<JoinTupleBase, In1>::value, JoinTuplePtr>::type findCorrectJoinTuple
    (std::vector<std::string> &typeList, std::vector<int> &whereEveryoneGoes);

template<typename In1>
typename std::enable_if<!std::is_base_of<JoinTupleBase, In1>::value, JoinTuplePtr>::type findCorrectJoinTuple
    (std::vector<std::string> &typeList, std::vector<int> &whereEveryoneGoes) {

  // we must always have one type...
  JoinTuplePtr returnVal;
  std::string in1Name = getTypeName<Handle<In1>>();
  std::cout << "in1Name=" << in1Name << std::endl;
  int in1Pos = findType(in1Name, typeList);

  if (in1Pos != -1) {
    whereEveryoneGoes.push_back(in1Pos);
    typeList[in1Pos] = in1Name;
    return std::make_shared<JoinSingleton<JoinTuple<In1, char[0]>>>();
  } else {
    std::cout << "Why did we not find a type?\n";
    exit(1);
  }
}

template<typename In1>
typename std::enable_if<std::is_base_of<JoinTupleBase, In1>::value, JoinTuplePtr>::type findCorrectJoinTuple
    (std::vector<std::string> &typeList, std::vector<int> &whereEveryoneGoes) {
  return std::make_shared<JoinSingleton<In1>>();
}

template<typename In1, typename ...Rest>
typename std::enable_if<sizeof ...(Rest) != 0 && !std::is_base_of<JoinTupleBase, In1>::value,
                        JoinTuplePtr>::type findCorrectJoinTuple
    (std::vector<std::string> &typeList, std::vector<int> &whereEveryoneGoes) {

  JoinTuplePtr returnVal;
  std::string in1Name = getTypeName<Handle<In1>>();
  std::cout << "in1Name =" << in1Name << std::endl;
  int in1Pos = findType(in1Name, typeList);

  if (in1Pos != -1) {
    returnVal = findCorrectJoinTuple<JoinTuple<In1, char[0]>, Rest...>(typeList, whereEveryoneGoes);
    whereEveryoneGoes.push_back(in1Pos);
    typeList[in1Pos] = in1Name;
  } else {
    returnVal = findCorrectJoinTuple<Rest...>(typeList, whereEveryoneGoes);
  }

  return returnVal;
}

template<typename In1, typename In2, typename ...Rest>
typename std::enable_if<std::is_base_of<JoinTupleBase, In1>::value, JoinTuplePtr>::type findCorrectJoinTuple
    (std::vector<std::string> &typeList, std::vector<int> &whereEveryoneGoes) {

  JoinTuplePtr returnVal;
  std::string in2Name = getTypeName<Handle<In2>>();
  int in2Pos = findType(in2Name, typeList);

  if (in2Pos != -1) {
    returnVal = findCorrectJoinTuple<JoinTuple<In2, In1>, Rest...>(typeList, whereEveryoneGoes);
    whereEveryoneGoes.push_back(in2Pos);
    typeList[in2Pos] = in2Name;
  } else {
    returnVal = findCorrectJoinTuple<In1, Rest...>(typeList, whereEveryoneGoes);
  }

  return returnVal;
}

}

#endif //PDB_JOINTUPLESINGLETON_H
