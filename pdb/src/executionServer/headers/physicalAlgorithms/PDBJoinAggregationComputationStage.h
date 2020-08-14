#pragma once

#include "PDBPhysicalAlgorithmStage.h"
#include "JoinCompBase.h"

namespace pdb {

class PDBJoinAggregationComputationStage : public PDBPhysicalAlgorithmStage {
 public:

  PDBJoinAggregationComputationStage(const PDBSinkPageSetSpec &sink,
                                     const PDBSinkPageSetSpec &preaggIntermediate,
                                     const Vector<PDBSourceSpec> &sources,
                                     const String &final_tuple_set,
                                     const Vector<pdb::Handle<PDBSourcePageSetSpec>> &secondary_sources,
                                     const Vector<PDBSetObject> &sets_to_materialize,
                                     const String &join_tuple_set,
                                     const PDBSourcePageSetSpec &left_join_source,
                                     const PDBSourcePageSetSpec &right_join_source,
                                     const PDBSinkPageSetSpec &intermediate_sink,
                                     const Vector<PDBSourceSpec> &right_sources);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManager> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManager> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManager> &storage) override;

 private:

  /**
   *
   * @param storage
   * @param idx
   * @return
   */
  PDBAbstractPageSetPtr getRightSourcePageSet(const std::shared_ptr<pdb::PDBStorageManager> &storage,
                                              size_t idx);

  /**
   *
   * @param catalogClient
   * @param idx
   * @return
   */
  pdb::SourceSetArgPtr getRightSourceSetArg(const std::shared_ptr<pdb::PDBCatalogClient> &catalogClient,
                                            size_t idx);

  /**
   * Returns the join computation in this join aggregation pipeline
   * @return the join comp
   */
  JoinCompBase *getJoinComp(const LogicalPlanPtr &logicalPlan);

  // the join tuple set
  const pdb::String &joinTupleSet;

  //
  const PDBSinkPageSetSpec &preaggIntermediate;

  //
  const PDBSourcePageSetSpec &leftJoinSource;

  //
  const PDBSourcePageSetSpec &rightJoinSource;

  //
  const PDBSinkPageSetSpec &intermediateSink;

  // The sources of the right side of the merged pipeline
  const pdb::Vector<PDBSourceSpec> &rightSources;
};

}