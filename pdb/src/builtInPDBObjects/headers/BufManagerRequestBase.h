#pragma once

#include <atomic>
#include "Object.h"

namespace pdb {

#ifdef DEBUG_BUFFER_MANAGER

class BufManagerRequestBase : public pdb::Object {
public:

  // the default constructor
  BufManagerRequestBase() {

    // init the id
    currentID = lastID++;
  }

  // the copy constructor
  BufManagerRequestBase(const BufManagerRequestBase &obj) {
    this->currentID = obj.currentID;
  }

  // the current id of the request
  std::uint64_t currentID;

private:

  static std::atomic<std::uint64_t> lastID;
};

#else

using BufManagerRequestBase = pdb::Object;

#endif

}
