#pragma once

#include "SetWriter.h"

namespace pdb {

  class IntWriter : public  pdb::SetWriter<int>{

   public:

    ENABLE_DEEP_COPY

    IntWriter() = default;

    // below constructor is not required, but if we do not call setOutputSet() here, we must call
    // setOutputSet() later to set the output set
    IntWriter(const std::string& dbName, const std::string& setName) {
      this->setOutputSet(dbName, setName);
    }
  };

}