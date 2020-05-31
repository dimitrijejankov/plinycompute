#pragma once

#include "PDBPhysicalAlgorithmStage.h"
#include "PDBPageNetworkSender.h"
#include "PDBPageSelfReceiver.h"
#include "PDBJoinAggregationState.h"

namespace pdb {

class PDBJoinAggregationKeyStage : public PDBPhysicalAlgorithmStage {
public:

  PDBJoinAggregationKeyStage(const PDBSinkPageSetSpec &sink,
                             const Vector<PDBSourceSpec> &sources,
                             const String &final_tuple_set,
                             const Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondary_sources,
                             const Vector<PDBSetObject> &sets_to_materialize,
                             const String &left_input_tuple_set,
                             const  String &right_input_tuple_set,
                             const String &join_tuple_set,
                             const PDBSinkPageSetSpec &lhs_key_sink,
                             const PDBSinkPageSetSpec &rhs_key_sink,
                             const PDBSinkPageSetSpec &join_agg_key_sink,
                             const Vector<PDBSourceSpec> &right_sources);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state) override;

  // the lhs input set to the join aggregation pipeline
  const pdb::String &leftInputTupleSet;

  // the rhs input set to the join aggregation pipeline
  const pdb::String &rightInputTupleSet;

  // the join tuple set
  const pdb::String &joinTupleSet;

  // the page set we use to store the result of the left key pipeline
  const PDBSinkPageSetSpec &lhsKeySink;

  // the page set we use to store the result of the right key pipeline
  const PDBSinkPageSetSpec &rhsKeySink;

  // the final sink where we store the keys of the aggregation
  const PDBSinkPageSetSpec &joinAggKeySink;

  // The sources of the right side of the merged pipeline
  const pdb::Vector<PDBSourceSpec> &rightSources;

  // this sends the plan
  static bool sendPlan(const std::string &ip, int32_t port,
                       const PDBBufferManagerInterfacePtr &mgr,
                       const Handle<pdb::ExJob> &job,
                       const PDBPhysicalAlgorithmStatePtr &state);

  // recieve the plan
  static bool receivePlan(const PDBBufferManagerInterfacePtr &mgr,
                          const Handle<pdb::ExJob> &job,
                          const PDBPhysicalAlgorithmStatePtr &state);

 private:

  static bool runLead(const Handle<pdb::ExJob> &job,
                      const PDBPhysicalAlgorithmStatePtr &state,
                      const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                      const std::string &error);

  static bool runFollower(const Handle<pdb::ExJob> &job,
                          const PDBPhysicalAlgorithmStatePtr &state,
                          const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                          const std::string &error);

  static pdb::SourceSetArgPtr getKeySourceSetArg(std::shared_ptr<pdb::PDBCatalogClient> &catalogClient,
                                                 const pdb::Vector<PDBSourceSpec> &sources,
                                                 size_t idx);


  static PDBAbstractPageSetPtr getKeySourcePageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                   size_t idx,
                                                   const pdb::Vector<PDBSourceSpec> &srcs);

  static PDBAbstractPageSetPtr getFetchingPageSet(const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                                  size_t idx,
                                                  const pdb::Vector<PDBSourceSpec> &srcs,
                                                  const std::string &ip,
                                                  int32_t port);
};

}