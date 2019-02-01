

/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/

#ifndef PAGE_H
#define PAGE_H

#include <memory>
#include "PDBSet.h"
#include <string>
#include <mutex>
#include <atomic>

// this is the smallest mini-page size that we can have
#define MIN_PAGE_SIZE 8u

// create a smart pointer for pages
using namespace std;

namespace pdb {

class PDBPage;
typedef shared_ptr <PDBPage> PDBPagePtr;
typedef weak_ptr <PDBPage> PDBPageWeakPtr;

struct PDBPageInfo {
  int64_t startPos = 0;
  int64_t numBytes = 0;
};

enum PDBPageStatus {

  PDB_PAGE_LOADING,
  PDB_PAGE_LOADED,
  PDB_PAGE_UNLOADING,
  PDB_PAGE_FREEZING,
  PDB_PAGE_NOT_LOADED,
};

// forward definition to handle circular dependencies
class PDBStorageManagerInterface;

class PDBPage {

 public:

  // tells us whether the page is pinned and/or dirty
  bool isPinned ();
  bool isDirty ();

  // get the bytes for this page.  This should ONLY be called on a pinned page, or the bytes
  // can non-deterministically be made invalid, if the page is paged out by another thread
  void *getBytes ();

  // an anonymous page is one that is used for temporary storage, and NOT for data that is
  // to be permanently written to disk
  // for a non-anymous page, tells us the location of the page in the associated data set
  // (0, 1, 2, 3, ..., up until the last page in the set)
  size_t whichPage ();

  // gets info about the set that this page is associated with
  PDBSetPtr getSet ();

  // unpins a page, (potentially) freeing its memory.  After this call, the pointr returned
  // by getBytes () may be unreliable as the page may be written to disk and the memory
  // reused at any time
  void unpin ();

  // freeze the size of the page at at most numBytes.  This can ONLY be called before the
  // first time that the page is unpinned
  void freezeSize (size_t numBytes);

  // pins a page.  At creation (when they are created by the buffer manager) all pages are
  // pinned.  If a page is unpinned, it can be pinned again later via a call to ths method.
  // Note that after re-pinning, getBytes () must be called as the location of the page in
  // RAM may have changed
  void repin ();

  // create a page
  explicit PDBPage (PDBStorageManagerInterface &);

 protected:

  // a bunch of setters (and some getters) that should ONLY be called by thr storage manager
  void setSet (PDBSetPtr);
  unsigned numRefs ();
  PDBPageInfo &getLocation ();
  PDBPageStatus &getStatus();
  void setPageNum (size_t);
  void setAnonymous (bool);
  bool isAnonymous ();
  void setBytes (void *);
  bool sizeIsFrozen ();
  void setPinned ();
  void freezeSize ();
  void setUnpinned ();
  void setDirty ();
  void setClean ();
  void setMe (PDBPagePtr toMe);
  void incRefCount ();
  void decRefCount ();

  // a pointer to the raw bytes
  void *bytes;

  // the status of the page
  PDBPageStatus status;

  // these are all pretty self-explanatory!
  bool pinned = false;
  std::atomic_bool dirty;
  unsigned refCount = 0;
  size_t pageNum = 0;
  bool isAnon = true;
  bool sizeFrozen = false;
  PDBPageInfo location;
  PDBSetPtr whichSet = nullptr;
  PDBPageWeakPtr me;

  // pointer to the parent buffer manager
  PDBStorageManagerInterface& parent;

  // the mutex to lock the page
  std::mutex lk;

  friend class PDBPageHandleBase;
  friend class PDBStorageManagerImpl;
  friend class PDBStorageManagerFrontEnd;

  template <class T>
  friend class PDBStorageManagerBackEnd;
};

}


#endif

