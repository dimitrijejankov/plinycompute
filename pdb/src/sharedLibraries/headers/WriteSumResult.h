#pragma once

#include <SumResult.h>
#include <SetWriter.h>
#include <LambdaCreationFunctions.h>
#include <VectorTupleSetIterator.h>
#include <VectorSink.h>

namespace pdb {

class WriteSumResult : public pdb::SetWriter<SumResult> {

 public:

  ENABLE_DEEP_COPY

  WriteSumResult() = default;

  WriteSumResult(std::string dbName, std::string setName) {
    this->setOutputSet(std::move(dbName), std::move(setName));
  }
};

}