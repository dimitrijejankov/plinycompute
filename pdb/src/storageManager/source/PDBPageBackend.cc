

/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/

#ifndef PAGE_BACKEND_C
#define PAGE_BACKEND_C

#include <PDBPageBackend.h>
#include <PDBPage.h>
#include "PDBStorageManagerImpl.h"
#include "PDBSet.h"

namespace pdb {

void PDBPageBackend::incRefCount() {

  // lock the page so we can increment the reference count
  std::unique_lock<mutex> l(lk);

  // increment the reference count
  refCount++;

  // increment the reference count
  PDBPage::incRefCount();
}

void PDBPageBackend::decRefCount() {

  // lock the page so we can check the reference count, decrement it and free it if needed
  std::unique_lock<mutex> l(lk);

  // decrement the reference count
  PDBPage::decRefCount();
}

}

#endif

