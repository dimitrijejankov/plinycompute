//
// Created by dimitrije on 10/16/18.
//

#ifndef PDB_STOFREEZESIZEREQUEST_H
#define PDB_STOFREEZESIZEREQUEST_H

// PRELOAD %BufFreezeSizeRequest%

#include "PDBString.h"
#include "PDBSet.h"
#include "BufManagerRequestBase.h"

namespace pdb {

// request to get an anonymous page
class BufFreezeSizeRequest : public BufManagerRequestBase {

 public:

  BufFreezeSizeRequest(const PDBSetPtr &set, const size_t &pageNumber, size_t freezeSize)
      : isAnonymous(set == nullptr), pageNumber(pageNumber), freezeSize(freezeSize) {

    // is this an anonymous page if it is
    if(!isAnonymous) {
      databaseName = pdb::makeObject<pdb::String>(set->getDBName());
      setName = pdb::makeObject<pdb::String>(set->getSetName());
    }
  }

  BufFreezeSizeRequest(const std::string &setName, const std::string &dbName, const size_t &pageNumber, size_t freezeSize)
      : isAnonymous(false), pageNumber(pageNumber), freezeSize(freezeSize) {

    this->setName = pdb::makeObject<pdb::String>(setName);
    this->databaseName = pdb::makeObject<pdb::String>(dbName);
  }

  explicit BufFreezeSizeRequest(const pdb::Handle<BufFreezeSizeRequest>& copyMe) : BufManagerRequestBase(*copyMe){

    // copy stuff
    isAnonymous = copyMe->isAnonymous;
    databaseName = copyMe->databaseName;
    setName = copyMe->setName;
    pageNumber = copyMe->pageNumber;
    freezeSize = copyMe->freezeSize;
  }

  BufFreezeSizeRequest() = default;

  ~BufFreezeSizeRequest() = default;

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
