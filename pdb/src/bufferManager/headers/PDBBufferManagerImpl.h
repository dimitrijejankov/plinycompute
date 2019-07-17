

#ifndef STORAGE_MGR_H
#define STORAGE_MGR_H

#include "PDBBufferManagerCheckLRU.h"
#include "PDBPage.h"
#include "PDBPageHandle.h"
#include "PDBSet.h"
#include "PDBPageCompare.h"
#include "PDBSetCompare.h"
#include "PDBSharedMemory.h"
#include "PDBBufferManagerInterface.h"
#include "NodeConfig.h"

#include <map>
#include <memory>
#include <condition_variable>
#include <queue>
#include <set>

/**
 * This is the class that implements PC's per-node storage manager.
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
 * Because the actual pages are variable sized, we don't figure out where in a file a non-anonymous page
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

using namespace std;

namespace pdb {

class PDBBufferManagerImpl : public PDBBufferManagerInterface {

 public:

  /**
   * we need the default constructor for our tests
   */
  PDBBufferManagerImpl() = default;

  /**
   * initializes the storage manager using the node configuration
   * it will check if the node already contains a metadata file if it does it will initialize the
   * storage manager using the metadata, otherwise it will use the page and memory info of the node configuration
   * to create a new storage
   * @param config - the configuration of the node we need this so we can figure out where to put the data and metadata
   */
  explicit PDBBufferManagerImpl(pdb::NodeConfigPtr config);

  /**
   * Simply loop through and write back any dirty pages.
   */
  ~PDBBufferManagerImpl() override;

  /**
   * Initialize a storage manager.  Anonymous pages will be written to tempFile.  Use the given pageSize.
   * The number of pages available to buffer data in RAM is numPages.  All meta-data is written to
   * metaDataFile.  All files to hold database files are written to the directory storageLoc
   * @param tempFile - path to the temporary file
   * @param pageSize - The size of a page in bytes
   * @param numPages - number physical pages to allocate
   * @param metaDataFile - the file where we store the metadata of the buffer manager
   * @param storageLocIn - path to the folder where we store the set data
   */
  void initialize(std::string tempFile, size_t pageSize, size_t numPages, std::string metaDataFile, std::string storageLocIn);

  /**
   * initialize the storage manager using the file metaDataFile
   * @param metaDataFile
   */
  void initialize(std::string metaDataFile);

  /**
   * gets the i^th page in the table whichSet... note that if the page
   * is currently being used (that is, the page is current buffered) a handle
   * to that already-buffered page should be returned
   *
   * Under the hood, the storage manager first makes sure that it has a file
   * descriptor for the file storing the page's set.  It then checks to see
   * if the page already exists.  It it does, we just return it.  If the page
   * does not already exist, we see if we have ever created the page and
   * written it back before.  If we have, we go to the disk location for the
   * page and read it in.  If we have not, we simply get an empty set of
   * bytes to store the page and return that.
   * @param whichSet - this is the set identifier to which the page belongs to (databaseName, setName)
   * @param i - the i-th page of the set
   * @return - a page handle to the requested page, it is guaranteed to be pinned
   */
  PDBPageHandle getPage(PDBSetPtr whichSet, uint64_t i) override;

  /**
   * gets a temporary page that will no longer exist (1) after the buffer manager
   * has been destroyed, or (2) there are no more references to it anywhere in the
   * program.  Typically such a temporary page will be used as buffer memory.
   * since it is just a temp page, it is not associated with any particular
   * set.  On creation, the page is pinned until it is unpinned.
   *
   * Under the hood, this simply finds a mini-page to store the page on (kicking
   * existing data out of the buffer if necessary)
   * @return - a page handle to an anonymous page of the maximum page size, it is guaranteed to be pinned
   */
  PDBPageHandle getPage() override;

  /**
   * gets a temporary page that is at least minBytes in size
   * @param minBytes - the minimum bytes the page needs to have
   * @return - a page handle to an anonymous page, it is guaranteed to be pinned and have a size of at least minBytes
   */
  PDBPageHandle getPage(size_t minBytes) override;

  /**
   * Returns the maximum page size this buffer manager can give.
   * @return - the maximum page size
   */
  size_t getMaxPageSize() override;

  /**
   * the storage manager does not have any server functionalities, they will be defined in the frontend (makes testing easier)
   * @param forMe - this is a reference to the PDBServer for which we want to register the handles for
   */
  void registerHandlers(PDBServer &forMe) override {};

  /**
   * clears all the info about a particular set
   * @param set - the set we want to clear
   */
  void clearSet(const PDBSetPtr &set);

protected:

  /**
   * Checks if the file of the set is open if it is not then just open it, if it does not exist create it..
   * @param whichSet - the set we want to check the file for.
   */
  void checkIfOpen(PDBSetPtr &whichSet);

  /**
   * Returns the file descriptor for a particular set
   * @param whichSet - the set we are looking up the file descriptor
   * @return - the file descriptor
   */
  int getFileDescriptor(const PDBSetPtr &whichSet);

  /**
   * Returns the nearest log of page size that can accommodate the requested number of bytes
   * @param numBytes - the number of bytes that needs to the be on that page
   * @return - the value
   */
  size_t getLogPageSize(size_t numBytes);

  /**
   * "registers" a min-page.  That is, do record-keeping so that we can link the mini-page
   * to the full page that it is located on top of.  Since this is called when a page is created
   * or read back from disk, it calls "pinParent" to make sure that the parent (full) page cannot be
   * written out
   * @param registerMe - the page we want to register
   */
  void registerMiniPage(const PDBPagePtr& registerMe);

  /**
   * this creates additional mini-pages of size MIN_PAGE_SIZE * 2^whichSize.  To do this, it
   * looks at the full page with the largest LRU number.  It then looks through all of the
   * mini-pages that have been allocated on that full page, and frees each of them.  To free
   * such a page, there are two cases: the page to be freed is anonymous, or it is not.  If it
   * is anonymous, then if it is dirty, we get a spot for it in the temp file and kick it out.
   * If the page is not anonymous, it is written back (it must already have a spot to be
   * written to, because it has to have been unpinned) and then if there are no references to
   * it, it is destroyed.
   * @param whichSize
   * @param lock
   */
  void createAdditionalMiniPages(int64_t whichSize, unique_lock<mutex> &lock);

  /**
   * tell the buffer manager that the given page can be truncated at the indicated size
   * @param me - the page we want to freeze
   * @param numBytes - the number of bytes we want to freeze it to. (rounded up to the nearest base 2 exponent)
   */
  void freezeSize(PDBPagePtr me, size_t numBytes) override;

  /**
   * The same as @see freezeSize but it takes in the unique_lock holding the locked mutex of the buffer manager
   * @param me - the page we want to freeze
   * @param numBytes - the number of bytes we want to freeze it to. (rounded up to the nearest base 2 exponent)
   * @param lock - the lock holding the locked mutex of the buffer manager
   */
  void freezeSize(PDBPagePtr me, size_t numBytes, unique_lock<mutex> &lock);

  /**
   * unpin the page.  This freezes the size of the page (because now the page is read-only)
   * and then decrements the number of pinned pages on this pages' full parent page.  If this
   * page is not anonymous, we determine where its actual location on disk will be (for an
   * anonymous page, we wait until the page has to be written back to determine its location,
   * because unlike non-anonymous pages, anonymous pages will often never make it to disk)
   * @param me - the page we want to unpin
   */
  void unpin(PDBPagePtr me) override;

  /**
   * The same as @see unpin but it takes in the unique_lock holding the locked mutex of the buffer manager
   * @param me - the page we want to unpin
   * @param lock - the lock holding the locked mutex of the buffer manager
   */
  void unpin(PDBPagePtr me, unique_lock<mutex> &lock);

  /**
   * pins the page that is the parent of a mini-page.  The "parent" is the page that contains
   * the physical bits for the mini-page.  To pin the parent, we first determine the parent,
   * then we increment the number of pinned pages in the parent.  If the parent is not currently
   * pinned, we remove it from the LRU queue (its current LRU number is the negative of the
   * number of pinned pages in this case)
   * @param me - the page whose parent page (physical page) we want to pin
   * that were removed out of use when the parent was, inserted into the LRU.
   */
  void pinParent(const PDBPagePtr& me);

  /**
   * repins a page (it is called "repin" because by definition, each page is pinned upon
   * creation, so every page has been pinned at least once).  To repin, if the page is already
   * in RAM, we just pin the page, and then pin the page's parent.  If it is not in RAM, then
   * if it is not in RAM, then we get a mini-page to store this guy, read it in, register the
   * mini page he is written on (this allows the parent page to be aware that the mini-page
   * is located on top of him, so he can't be kicked out while the mini-page is pinned), and
   * then note that this guy is now pinned
   * @param me - the page we want to repin
   */
  void repin(PDBPagePtr me) override;

  /**
   * The same as @see repin but it takes in the unique_lock holding the locked mutex of the buffer manager
   * @param me - the page we want to repin
   * @param lock - the lock holding the locked mutex of the buffer manager
   */
  void repin(PDBPagePtr me, unique_lock<mutex> &lock);

  /**
   * this is called when there are no more external references to an anonymous page, and so
   * it can be destroyed.  To do this, we first unpin it (if it is pinned) and then remove it
   * from its parent's list of constituent pages.
   * @param me - the anonymous page we want to free
   */
  void freeAnonymousPage(PDBPagePtr me) override;

  /**
   * this is called when there are zero external references to a page belonging to a set.  We remove all traces
   * of the page from the system, as long as the page is not being buffered in RAM (if it is,
   * then the page may be removed later if its parent page is recycled)
   * @param me - the set page we want to free
   */
  void downToZeroReferences(PDBPagePtr me) override;

  /**
   * So since we don't lock the buffer manager every time we update the reference count on a page but rather a
   * lock local to the page, it can happen that the request to freeAnonymousPage or downToZeroReferences
   * is stale and we ended up doing something else with the page in the mean time. This method checks if something
   * happened to the page in the mean time.
   * @param me - the page we are supposed to remove
   * @return true if it is, false otherwise
   */
  bool isRemovalStillValid(PDBPagePtr me);

  /**
   * this method finds free memory for a page of the specified size
   * @param pageSize - the size of the page
   * @return - a void pointer pointing to a memory of exactly the specified size
   */
  void *getEmptyMemory(int64_t pageSize, unique_lock<mutex> &lock);

  /**
   * list of ALL of the page objects that are currently in existence
   */
  map<pair<PDBSetPtr, size_t>, PDBPagePtr, PDBPageCompare> allPages;

  /**
   * tells us, for each set, where each of the various pages are physically located.  The i^th entry in the
   * vector tells us where to find the i^th entry in the set
   */
  map<pair<PDBSetPtr, size_t>, PDBPageInfo, PDBPageCompare> pageLocations;

  /**
   * this tells us, for each set, the last used location in the file
   */
  map<PDBSetPtr, size_t, PDBSetCompare> endOfFiles;

  /**
   * this keeps the LRU numbers sorted so that we can quickly evict a parent page
   */
  set<pair<void *, size_t>, PDBBufferManagerCheckLRU> lastUsed;

  /**
   * tells us how many of the minipages constructed from each page are pinned if the long is a negative value, it gives us the LRU number
   */
  map<void *, long> numPinned;

  /**
   * lists the FDs for all of the files
   */
  map<PDBSetPtr, int, PDBSetCompare> fds;

  /**
   * all of the full pages that are currently not being used
   */
  vector<void *> emptyFullPages;

  /**
   * all of the mini-pages that make up a page
   */
  map<void *, vector<PDBPagePtr>> constituentPages;

  /**
   * all of the locations from which we are currently allocating minipages.  The first
   * entry in this vector is used to allocated minipages of size MIN_PAGE_SIZE, the
   * second minipages of size MIN_PAGE_SIZE * 2, and so on.  Each entry in the vector
   * is a pair containing a pointer to the full page that is being used to allocate
   * minipages, and a pointer to the next slot that we'll allocate from on the minipage
   */
  vector<vector<void *>> emptyMiniPages;

  /**
   * for a given full physical page maps all the mini pages it is split into but are not used
   * we have to keep track of this so that when we evict a page we know which pages to remove from the emptyMiniPages
   */
  map<void *, std::pair<vector<void *>, int64_t >> unusedMiniPages;

  /**
   * all of the positions in the temporary file that are currently not in use
   */
  vector<vector<int64_t>> availablePositions;

  /**
   * info about the shared memory of this storage manager contains the page size, number of pages and a pointer to
   * the shared memory
   */
  PDBSharedMemory sharedMemory{};

  /**
   * the time tick associated with the MRU page
   */
  long lastTimeTick = 1;

  /**
   * the last position in the temporary file
   */
  size_t lastTempPos = 0;

  /**
   * where we write the data
   */
  string tempFile;

  /**
   * the descriptor of the temporary file
   */
  int32_t tempFileFD = 0;

  /**
   * this is the log of pageSize / MIN_PAGE_SIZE
   */
  int64_t logOfPageSize = 0;

  /**
   * the location of the meta data file
   */
  string metaDataFile;

  /**
   * the location where data are written
   */
  string storageLoc;

  /**
   * whether the storage manager has been initialized
   */
  bool initialized = false;

  /**
   * locks the whole buffer manager
   */
  std::mutex m;

  /**
   * we use this conditional variable to avoid concurrency issues in the getPages for the set.
   * essentially if a page is being created or unloaded we use this conditional variable to block the thread that is
   * requesting that page
   */
  std::condition_variable pagesCV;

  /**
   * we use this conditional variable to block a thread that is requesting a minipage of the same size as the one
   * currently being created. This is done so that we don't accidentally evict two pages instead of just one
   * for more info @see createAdditionalMiniPages
   */
  std::condition_variable spaceCV;

  /**
   * false if it is not creating a conditional variable of a certain size, true if it is. The size is in log(pageSize).
   */
  std::vector<bool> isCreatingSpace;

  /**
   * this vector holds all the free page numbers we can assign to an anonymous page.
   */
  std::vector<uint64_t> freeAnonPageNumbers;

  /**
   * so we assign a unique number to each anonymous page so we can identify them, if we already assigned
   * a all available numbers to anonymous pages this number will tell us what is the next number we need to generate
   */
  int lastFreeAnonPageNumber = 0;

  /**
   * this locks the file descriptor structure
   */
  std::mutex fdLck;

  friend class PDBPage;
};

}

#endif


