//
// Created by dimitrije on 10/16/18.
//

#ifndef PDB_STOPINPAGEREQUEST_H
#define PDB_STOPINPAGEREQUEST_H

// PRELOAD %BufPinPageRequest%

#include "PDBString.h"
#include "PDBSet.h"
#include "BufManagerRequestBase.h"

namespace pdb {

// request to get an anonymous page
class BufPinPageRequest : public BufManagerRequestBase {

 public:

  BufPinPageRequest(const PDBSetPtr &set, const size_t &pageNumber)
      : isAnonymous(set == nullptr), pageNumber(pageNumber) {

    // is this an anonymous page if it is
    if(!isAnonymous) {
      databaseName = pdb::makeObject<pdb::String>(set->getDBName());
      setName = pdb::makeObject<pdb::String>(set->getSetName());
    }
  }

  BufPinPageRequest() = default;

  explicit BufPinPageRequest(const pdb::Handle<BufPinPageRequest>& copyMe) : BufManagerRequestBase(*copyMe) {

    // copy stuff
    isAnonymous = copyMe->isAnonymous;
    databaseName = copyMe->databaseName;
    setName = copyMe->setName;
    pageNumber = copyMe->pageNumber;
  }

  ~BufPinPageRequest() = default;

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

#endif //PDB_STOPINPAGEREQUEST_H
