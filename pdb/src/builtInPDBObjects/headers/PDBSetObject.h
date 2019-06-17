#pragma once

#include "Object.h"
#include "Handle.h"
#include "PDBString.h"

//  PRELOAD %PDBSetObject%

namespace pdb {

class PDBSetObject : public Object {
public:

  ENABLE_DEEP_COPY

  PDBSetObject(const String &database, const String &set) : database(database), set(set) {}

  PDBSetObject() = default;

  ~PDBSetObject() = default;

  /**
   * The name of the database
   */
  String database;

  /**
   * The name of the set
   */
  String set;

};

}