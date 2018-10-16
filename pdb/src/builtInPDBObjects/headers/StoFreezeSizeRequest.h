//
// Created by dimitrije on 10/16/18.
//

#ifndef PDB_STOFREEZESIZEREQUEST_H
#define PDB_STOFREEZESIZEREQUEST_H

// PRELOAD %StoFreezeSizeRequest%

#include "PDBString.h"
#include "PDBSet.h"

namespace pdb {

// request to get an anonymous page
class StoFreezeSizeRequest : public Object {

 public:

  StoFreezeSizeRequest(const PDBSetPtr &set, const size_t &pageNumber, size_t freezeSize)
      : isAnonymous(set == nullptr), pageNumber(pageNumber), freezeSize(freezeSize) {

    // is this an anonymous page if it is
    if(!isAnonymous) {
      databaseName = pdb::makeObject<pdb::String>(set->getDBName());
      setName = pdb::makeObject<pdb::String>(set->getSetName());
    }
  }

  StoFreezeSizeRequest() = default;

  ~StoFreezeSizeRequest() = default;

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

  /**
   * The size we want to freeze the page to
   */
  size_t freezeSize = 0;
};
}

#endif //PDB_STOFREEZESIZEREQUEST_H
