#pragma once

#include <atomic>
#include "Object.h"

namespace pdb {

class BufManagerRequestBase : public pdb::Object {
public:

  BufManagerRequestBase() {

    // init the id
    currentID = lastID++;
  }

  std::uint64_t currentID;

private:

  static std::atomic<std::uint64_t> lastID;
};

}
