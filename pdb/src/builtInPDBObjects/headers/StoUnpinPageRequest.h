//
// Created by dimitrije on 10/16/18.
//

#ifndef PDB_STOUNPINPAGEREQUEST_H
#define PDB_STOUNPINPAGEREQUEST_H

// PRELOAD %StoUnpinPageRequest%

#include "PDBString.h"
#include "PDBSet.h"

namespace pdb {

// request to get an anonymous page
class StoUnpinPageRequest : public Object {

public:

  StoUnpinPageRequest(const PDBSetPtr &set, const size_t &pageNumber)
      : isAnonymous(set == nullptr), pageNumber(pageNumber) {

    // is this an anonymous page if it is
    if(!isAnonymous) {
     databaseName = pdb::makeObject<pdb::String>(set->getDBName());
     setName = pdb::makeObject<pdb::String>(set->getSetName());
    }
  }

  StoUnpinPageRequest() = default;

  ~StoUnpinPageRequest() = default;

  ENABLE_DEEP_COPY;

  bool isAnonymous = false;

  /**
   * The database name
   */
  pdb::Handle<pdb::String> databaseName;

  /**
   * The set name
   */
  pdb::Handle<pdb::String> setName;

  /**
   * The page number
   */
  size_t pageNumber = 0;
};
}

#endif //PDB_STOUNPINPAGEREQUEST_H
