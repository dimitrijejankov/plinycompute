#pragma once

// PRELOAD %PDBJoin8Algorithm%

#include "PDBPhysicalAlgorithm.h"

namespace pdb {

  class PDBJoin8Algorithm : public PDBPhysicalAlgorithm {

    pdb::String in0;
    pdb::String out0;

    pdb::String in1;
    pdb::String out1;

    pdb::String in2;
    pdb::String out2;

    pdb::String in3;
    pdb::String out3;

    pdb::String in4;
    pdb::String out4;

    pdb::String in5;
    pdb::String out5;

    pdb::String in6;
    pdb::String out6;

    pdb::String in7;
    pdb::String out7;

    PDBSetObject source;
    PDBSetObject sink;

   public:
    PDBJoin8Algorithm();

    ENABLE_DEEP_COPY

    PDBJoin8Algorithm(const std::pair<std::string, std::string> &sourceSet,
                      const std::pair<std::string, std::string> &sinkSet,
                      const std::string &in0,
                      const std::string &out0,
                      const std::string &in1,
                      const std::string &out1,
                      const std::string &in2,
                      const std::string &out2,
                      const std::string &in3,
                      const std::string &out3,
                      const std::string &in4,
                      const std::string &out4,
                      const std::string &in5,
                      const std::string &out5,
                      const std::string &in6,
                      const std::string &out6,
                      const std::string &in7,
                      const std::string &out7);

    [[nodiscard]] PDBPhysicalAlgorithmStatePtr getInitialState(const pdb::Handle<pdb::ExJob> &job) const override;
    [[nodiscard]] vector<PDBPhysicalAlgorithmStagePtr> getStages() const override;
    [[nodiscard]] int32_t numStages() const override;
    PDBPhysicalAlgorithmType getAlgorithmType() override;
    PDBCatalogSetContainerType getOutputContainerType() override;

  };

}