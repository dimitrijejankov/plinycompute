

#ifndef STORAGE_MGR_IFC_H
#define STORAGE_MGR_IFC_H

#include <BufForwardPageRequest.h>
#include "PDBPage.h"
#include "PDBPageHandle.h"
#include "PDBSet.h"
#include "PDBCommunicator.h"
#include "ServerFunctionality.h"


/**
 * NOTE that this is an abstract class.  There are two instances of this class on any given node: one
 * that runs on the front end process, and one that runs on the back end process.  In general, the one
 * on the front end is doing all of the work of maitaining the buffer pool and reading/writing data
 * from/to disk.  The back end is mostly just a client for the front end.
 *
 * INTERFACE EXPLAINED FROM AN APPLICATION PROGRAMMERS POINT-OF-VIEW
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * There are two types of pages. Anonymous and non-anonymous. Anonymous pages don't correspond to a disk
 * page; they are pages used as RAM for temporary storage.  They remain in existence until all handles to
 * them are gone, at which time they disappear.  However, anonymous pages can be swapped out of RAM by
 * the storage manager, so it is possible to have more anonymous pages than can fit in the physical RAM
 * of the machine.
 *
 * Non-anonymous pages correspond to data that are stored on disk for later use.
 *
 * Pages by default are pageSize in bytes.  But, as an optimization, they can be smaller than this as
 * well.  There are two ways to get a smaller page.  One can simply create a small anonymous page by
 * calling getPage (maxBytes) which returns a page that may be as small as maxBytes in size.
 *
 * The other way is to call freezeSize (numBytes), which tells the storage manager that the page is never
 * going to use more than the first numBytes bytes on the page.  freezeSize () can be used on both
 * anonymous and non-anonymous pages.
 *
 * Beause the actual pages are variable sized, we don't figure out where in a file a non-anonymous page
 * is going to be written until it is unpinned (at which time its size cannot change).
 *
 * When a page is created, it is pinned.  It is kept in RAM until it is unpinned (a page can be unpinned
 * manually, or because there are no more handles in scope).  When a page is unpinned, it is assumed that
 * the page is read only, forever after.  Later it can be pinned again, but it is still read-only.
 *
 * At sometime before a page is unpinned, one can call freezeSize () on a handle to the page, informing
 * the system that not all of the bytes on the page are going to be used.  The system can then store only
 * the first, used part of the page, at great savings.
 *
 * When a large page is broken up into mini-pages, we keep track of how many mini-pages are pinned, and we
 * keep references to all of those mini-pages on the page.  As long as some mini-page on the page is pinned,
 * the entire page is pinned.  Once it contains no pinned mini-pages, it can potentially be re-cycled (and
 * all of the un-pinned mini-pages written back to disk).
 *
 * When a non-anonymous page is unpinned for the first time, we determine its true location on disk (pages
 * may not be located sequentially on disk, due to the fact that we have variable-sized pages, and we do
 * not know at the outset the actual number of bytes that will be used by a page).
 *
 * An anonymous page gets its location the first time that it is written out to disk.
 *
 * A page can be dirty or not dirty.  All pages are dirty at creation, but then once they are written out
 * to disk, they are clean forever more (by definition, a page needs to be unpinned to be written out to
 * disk, but once it is unpinned, it cannot be modified, so after it is written back, it can never be
 * modified again).
 */

namespace pdb {

class PDBBufferManagerInterface;
typedef shared_ptr<PDBBufferManagerInterface> PDBBufferManagerInterfacePtr;

class PDBBufferManagerInterface : public ServerFunctionality {

public:

  // gets the i^th page in the table whichSet... note that if the page
  // is currently being used (that is, the page is current buffered) a handle 
  // to that already-buffered page should be returned
  //
  // Under the hood, the storage manager first makes sure that it has a file
  // descriptor for the file storing the page's set.  It then checks to see
  // if the page already exists.  It it does, we just return it.  If the page
  // does not already exist, we see if we have ever created the page and
  // written it back before.  If we have, we go to the disk location for the
  // page and read it in.  If we have not, we simply get an empty set of 
  // bytes to store the page and return that.
  virtual PDBPageHandle getPage(PDBSetPtr whichSet, uint64_t i) = 0;

  // gets a temporary page that will no longer exist (1) after the buffer manager
  // has been destroyed, or (2) there are no more references to it anywhere in the
  // program.  Typically such a temporary page will be used as buffer memory.
  // since it is just a temp page, it is not associated with any particular 
  // set.  On creation, the page is pinned until it is unpinned.  
  // 
  // Under the hood, this simply finds a mini-page to store the page on (kicking
  // existing data out of the buffer if necessary)
  virtual PDBPageHandle getPage () = 0;

  // gets a temporary page that is at least minBytes in size
  virtual PDBPageHandle getPage (size_t minBytes) = 0;

  // gets the page size
  virtual size_t getMaxPageSize() = 0;

  // simply loop through and write back any dirty pages.  
  virtual ~PDBBufferManagerInterface () = default;

protected:

  // note that these methods are all going to be called by the PDBPage object when
  // an application programmer perfoms operations on the object.

  // this is called when there are no more external references to an anonymous page, and so
  // it can be destroyed.  To do this, we first unpin it (if it is pinned) and then remove it
  // from its parent's list of constituent pages.
  virtual void freeAnonymousPage (PDBPagePtr me) = 0;

  // this is called when there are zero external references to a page.  We remove all traces
  // of the page from the system, as long as the page is not being buffered in RAM (if it is,
  // then the page may be removed later if its parent page is recycled)
  virtual void downToZeroReferences (PDBPagePtr me) = 0;

  // tell the buffer manager that the given page can be truncated at the indcated size
  virtual void freezeSize (PDBPagePtr me, size_t numBytes) = 0;

  // unpin the page.  This freezes the size of the page (because now the page is read-only)
  // and then decrements the number of pinned pages on this pages' full parent page.  If this
  // page is not anonymous, we determine where its actual location on disk will be (for an
  // anonymous page, we wait until the page has to be written back to determine its location,
  // because unlike non-anonymous pages, anonymous pages will often never make it to disk)
  virtual void unpin (PDBPagePtr me) = 0;

  // repins a page (it is called "repin" because by definition, each page is pinned upon
  // creation, so every page has been pinned at least once).  To repin, if the page is already
  // in RAM, we just pin the page, and then pin the page's parent.  If it is not in RAM, then
  // if it is not in RAM, then we get a mini-page to store this guy, read it in, register the
  // mini page he is written on (this allows the parent page to be aware that the mini-page
  // is located on top of him, so he can't be kicked out while the mini-page is pinned), and
  // then note that this guy is now pinned
  virtual void repin (PDBPagePtr me) = 0;

#ifdef DEBUG_BUFFER_MANAGER
public:

  // we can supply the timestamp with these
  virtual void logGetPage(const PDBSetPtr &whichSet, uint64_t i, uint64_t timestamp) { logGetPage(whichSet, i); };
  virtual void logGetPage(size_t minBytes, uint64_t pageNumber, uint64_t timestamp) { logGetPage(minBytes, pageNumber); };
  virtual void logFreezeSize(const PDBSetPtr &setPtr, size_t pageNum, size_t numBytes, uint64_t timestamp) { logFreezeSize(setPtr, pageNum, numBytes); };
  virtual void logUnpin(const PDBSetPtr &setPtr, size_t pageNum, uint64_t timestamp) { logUnpin(setPtr, pageNum); };
  virtual void logRepin(const PDBSetPtr &setPtr, size_t pageNum, uint64_t timestamp) { logRepin(setPtr, pageNum); };
  virtual void logFreeAnonymousPage(uint64_t pageNumber, uint64_t timestamp) { logFreeAnonymousPage(pageNumber); };
  virtual void logDownToZeroReferences(const PDBSetPtr &setPtr, size_t pageNum, uint64_t timestamp) { logDownToZeroReferences(setPtr, pageNum); };
  virtual void logClearSet(const PDBSetPtr &set, uint64_t timestamp) { logClearSet(set); };


  // these are virtual so we can hijack the page methods
  virtual void logGetPage(const PDBSetPtr &whichSet, uint64_t i) {};
  virtual void logGetPage(size_t minBytes, uint64_t pageNumber) {};
  virtual void logFreezeSize(const PDBSetPtr &setPtr, size_t pageNum, size_t numBytes) {};
  virtual void logUnpin(const PDBSetPtr &setPtr, size_t pageNum) {};
  virtual void logRepin(const PDBSetPtr &setPtr, size_t pageNum) {};
  virtual void logFreeAnonymousPage(uint64_t pageNumber) {};
  virtual void logDownToZeroReferences(const PDBSetPtr &setPtr, size_t pageNum) {};
  virtual void logClearSet(const PDBSetPtr &set) {};
  virtual void logExpect(const Handle<BufForwardPageRequest> &result) {};
  virtual void logForward(const Handle<pdb::BufForwardPageRequest> &request) {};

  // we need to mark the expect page as override so we make it like that
  #define PDB_BACKEND_EXPECT_POSTFIX override

  // just so we can have a debug manager
  virtual PDBPageHandle expectPage(std::shared_ptr<PDBCommunicator> &communicator) { throw runtime_error("Can't call expect page on this object!"); };

#else
protected:

  // all of these are going to be optimized out
  static void logGetPage(const PDBSetPtr& whichSet, uint64_t i) {};
  static void logGetPage(size_t minBytes, uint64_t pageNumber) {};
  static void logFreezeSize(const PDBSetPtr &setPtr, size_t pageNum, size_t numBytes) {};
  static void logUnpin(const PDBSetPtr &setPtr, size_t pageNum) {};
  static void logRepin(const PDBSetPtr &setPtr, size_t pageNum) {};
  static void logFreeAnonymousPage(uint64_t pageNumber) {};
  static void logDownToZeroReferences(const PDBSetPtr &setPtr, size_t pageNum) {};
  static void logClearSet(const PDBSetPtr &set) {};
  static void logExpect(const Handle<BufForwardPageRequest> &result) {};
  static void logForward(const Handle<pdb::BufForwardPageRequest> &request) {};

  // we should not mark expectPage as override since it is not virtual
  #define PDB_BACKEND_EXPECT_POSTFIX

#endif

  // so that the PDBPage can access the above methods
  friend class PDBPage;
};

}


#endif


