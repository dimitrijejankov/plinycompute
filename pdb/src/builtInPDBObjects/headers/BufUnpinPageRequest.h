//
// Created by dimitrije on 10/16/18.
//

#ifndef PDB_STOUNPINPAGEREQUEST_H
#define PDB_STOUNPINPAGEREQUEST_H

// PRELOAD %BufUnpinPageRequest%

#include "PDBString.h"
#include "PDBSet.h"
#include "BufManagerRequestBase.h"

namespace pdb {

// request to get an anonymous page
class BufUnpinPageRequest : public BufManagerRequestBase {

public:

  BufUnpinPageRequest(const PDBSetPtr &set, const size_t &pageNumber, bool isDirty)
      : isAnonymous(set == nullptr), pageNumber(pageNumber), isDirty(isDirty) {

    // is this an anonymous page if it is
    if(!isAnonymous) {
     databaseName = pdb::makeObject<pdb::String>(set->getDBName());
     setName = pdb::makeObject<pdb::String>(set->getSetName());
    }
  }

  BufUnpinPageRequest() = default;

  explicit BufUnpinPageRequest(const pdb::Handle<BufUnpinPageRequest>& copyMe) : BufManagerRequestBase(*copyMe) {

    // copy stuff
    isAnonymous = copyMe->isAnonymous;
    isDirty = copyMe->isDirty;
    databaseName = copyMe->databaseName;
    setName = copyMe->setName;
    pageNumber = copyMe->pageNumber;
  }

  ~BufUnpinPageRequest() = default;

  ENABLE_DEEP_COPY;

  /**
   * is the page we are unpinning dirty
   */
  bool isAnonymous = false;

  /**
   * is the page dirty
   */
  bool isDirty = false;

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
