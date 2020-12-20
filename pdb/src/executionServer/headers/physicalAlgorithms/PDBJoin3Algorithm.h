#pragma once

// PRELOAD %PDBJoin3Algorithm%

#include "PDBPhysicalAlgorithm.h"

namespace pdb {

  class PDBJoin3Algorithm : public PDBPhysicalAlgorithm {

    pdb::String in0;
    pdb::String out0;

    pdb::String in1;
    pdb::String out1;

    pdb::String in2;
    pdb::String out2;

    pdb::String out3;
    pdb::String final;

    PDBSetObject source0;
    PDBSetObject source1;
    PDBSetObject source2;
    PDBSetObject sink;

   public:
    PDBJoin3Algorithm();

    ENABLE_DEEP_COPY

    PDBJoin3Algorithm(const std::pair<std::string, std::string> &sourceSet0,
                      const std::pair<std::string, std::string> &sourceSet1,
                      const std::pair<std::string, std::string> &sourceSet2,
                      const std::pair<std::string, std::string> &sinkSet,
                      const std::string &in0,
                      const std::string &out0,
                      const std::string &in1,
                      const std::string &out1,
                      const std::string &in2,
                      const std::string &out2,
                      const std::string &out3,
                      const std::string &final);

    [[nodiscard]] PDBPhysicalAlgorithmStatePtr getInitialState(const pdb::Handle<pdb::ExJob> &job) const override;
    PDBPhysicalAlgorithmStagePtr getNextStage(const PDBPhysicalAlgorithmStatePtr &state) override;
    [[nodiscard]] int32_t numStages() const override;
    PDBPhysicalAlgorithmType getAlgorithmType() override;
    PDBCatalogSetContainerType getOutputContainerType() override;

  };

}