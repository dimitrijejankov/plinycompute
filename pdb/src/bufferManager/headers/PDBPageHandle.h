

/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/


#ifndef PAGE_HANDLE_H
#define PAGE_HANDLE_H

#include <memory>
#include "PDBPage.h"
#include "PDBSet.h"
#include <string>

namespace pdb {

// page handles are basically smart pointers
using namespace std;
class PDBPageHandleBase;
typedef shared_ptr<PDBPageHandleBase> PDBPageHandle;

class PDBPageHandleBase {

 public:

  void freezeSize(size_t numBytes) {
    return page->freezeSize(numBytes);
  }

  // tells us whether the page is pinned and/or dirty
  bool isPinned() {
    return page->isPinned();
  }

  bool isDirty() {
    return page->isDirty();
  }

  void setDirty() {
    page->setDirty();
  }

  // get the bytes for this page.  This should ONLY be called on a pinned page, or the bytes
  // can non-deterministically be made invalid, if the page is paged out by another thread
  void *getBytes() {
    return page->getBytes();
  }

  // for a non-anymous page, tells us the location of the page in the associated data set
  // (0, 1, 2, 3, ..., up until the last page in the set)
  size_t whichPage() {
    return page->whichPage();
  }

  // gets info about the set that this page is associated with
  PDBSetPtr getSet() {
    return page->getSet();
  }

  // unpins a page, (potentially) freeing its memory.  After this call, the pointr returned
  // by getBytes () may be unreliable as the page may be written to disk and the memory
  // reused at any time
  void unpin() {
    page->unpin();
  }

  // pins a page.  At creation (when they are created by the buffer manager) all pages are
  // pinned.  If a page is unpinned, it can be pinned again later via a call to ths method.
  // Note that after re-pinning, getBytes () must be called as the location of the page in
  // RAM may have changed
  void repin() {
    page->repin();
  }

  // returns the size of the page. If this is frozen it will return the frozen size, if not it will return
  size_t getSize() {
    return page->getSize();
  }

  // at destruction, we simply reduce the reference count of the page
  ~PDBPageHandleBase() {
    page->decRefCount();
  }

  // and at creation, increase the ref count
  explicit PDBPageHandleBase(PDBPagePtr &useMe) {
    page = useMe;
    page->incRefCount();
  }

 private:

  PDBPagePtr page;

  friend class PDBBufferManagerImpl;
  friend class PDBBufferManagerFrontEnd;

  template <class T>
  friend class PDBBufferManagerBackEnd;
  friend class PDBBufferManagerDebugBackEnd;
};

}

#endif

