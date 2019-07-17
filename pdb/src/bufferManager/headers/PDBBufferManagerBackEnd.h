/**
 * This is the part of the storage manager that is running in the back end.
 * There are two storage managers running on each machine: one on the front
 * end and one on the back end.
 *
 * Note that all communication via the front end and the back end storage managers
 * happens using the ServerFunctionality interface that both implement (though
 * the only time that the front end storage manager directly contacts the back end
 * is on startup, to send any necessary initialization information).  The ONE SUPER
 * IMPORTANT exception to this is that NO DATA on any page is ever sent over a
 * PDBCommunicator object.  Rather, when a page is requested by the back end or
 * a page is sent from the front end to the back end, the page is allocated into
 * the buffer pool, which is memory shared by the back end and the front end.  What
 * is actually sent over the PDBCommunicator is only a pointer into the shared
 * memory.
 *
 * The end doesn't do much work.  All it does is some internal bookkepping
 * to manage the pages it has received from the front end.  It does not actually buffer
 * anything; all of the buffering happens in the front end.  It basically just forwards
 * requests from the pages to the front end.
 */

#ifndef BE_STORAGE_MGR_H
#define BE_STORAGE_MGR_H

#include <PDBCommunicator.h>
#include <PDBServer.h>
#include "PDBSharedMemory.h"
#include <HeapRequest.h>
#include <condition_variable>
#include "PDBBufferManagerInterface.h"
#include "PDBPageCompare.h"

// this is needed so we can declare friend tests here
#include <gtest/gtest_prod.h>

namespace pdb {

#ifndef DEBUG_BUFFER_MANAGER

// all regular
template <class T>
class PDBBufferManagerBackEnd;
using PDBBufferManagerBackEndPtr = std::shared_ptr<PDBBufferManagerBackEnd<RequestFactory>>;
using PDBBufferManagerBackEndImpl = PDBBufferManagerBackEnd<RequestFactory>;

#endif

template <class T>
class PDBBufferManagerBackEnd : public PDBBufferManagerInterface {

public:

  explicit PDBBufferManagerBackEnd(const PDBSharedMemory &sharedMemory);

  ~PDBBufferManagerBackEnd() override = default;

  /**
   * Returns a handle to an anonymous page with the maximum page size.
   * Calling this method is thread safe.
   * @return - the handle to an anonymous page
   */
  PDBPageHandle getPage() override;

  /**
   * Returns a handle to a page of the specified set with the specified index.
   * @param whichSet -
   * @param i -
   * @return
   */
  PDBPageHandle getPage(PDBSetPtr whichSet, uint64_t i) override;

  /**
   * Returns a page that will be guaranteed to have at least minBytes size
   * Calling this method is thread safe.
   * @param minBytes - the minimum bytes required
   * @return - the handle to an anonymous page
   */
  PDBPageHandle getPage(size_t minBytes) override;

  /**
   *
   * @param communicator
   * @return
   */
  PDBPageHandle expectPage(std::shared_ptr<PDBCommunicator> &communicator) PDB_BACKEND_EXPECT_POSTFIX;


  /**
   * Returns the maximum page size as set in the configuration
   * @return - the value
   */
  size_t getMaxPageSize() override;

  void registerHandlers(PDBServer &forMe) override;

protected:


  /**
   * This method is called by the @see pdb::PDBPage when the reference count falls to zero and the page is anonymous.
   * The method will send a request to the frontend to free the anonymous page.
   * It is free to call without any locking since anonymous pages are not kept in the @see allPages map.
   * @param me - the page we want to free
   */
  void freeAnonymousPage(PDBPagePtr me) override;

  /**
   * This method is called by the @see pdb::PDBPage when the reference count falls to zero and the page belongs to a set.
   * It will send a request to the frontend to return the page back to it. The page might not get unpinned depending on
   * whether it is used on the frontend.
   * Before calling this method the page needs to be locked via the lock method
   * @param me - the page we want to return to the frontend.
   */
  void downToZeroReferences(PDBPagePtr me) override;

  /**
   * This method sends a request to the frontend to freeze the size of a page to the specified size,
   * it is called from the @see pdb::PDBPage that is called from the public method of @see pdb::PDBPageHandle::freezeSize
   * There is no locking on this method, the user has to ensure that it is called in a thread.
   * @param me - the page we want to freeze the size
   * @param numBytes - the size we want to freeze to
   */
  void freezeSize(PDBPagePtr me, size_t numBytes) override;

  /**
   * This method sends a request to the frontend to unpin a page,
   * it is called from the @see pdb::PDBPage that is called from the public method of @see pdb::PDBPageHandle::unpin
   * There is no locking on this method, the user has to ensure that it is called in a thread.
   * @param me
   */
  void unpin(PDBPagePtr me) override;

  /**
   * This method sends a request to the frontend to repin a page,
   * it is called from the @see pdb::PDBPage that is called from the public method of @see pdb::PDBPageHandle::unpin
   * There is no locking on this method, the user has to ensure that it is called in a thread.
   * @param me
   */
  void repin(PDBPagePtr me) override;

  // conditional variable to prevent multiple pages at the same time
  std::condition_variable cv;

  // mutex to lock the pages
  std::mutex m;

  // this is where we store all the backend pages
  map <pair <PDBSetPtr, size_t>, PDBPagePtr, PDBPageCompare> allPages;

  // the shared memory with the frontend
  PDBSharedMemory sharedMemory {};

  // Logger to debug information
  PDBLoggerPtr myLogger;

  // mark the tests for the backend
  FRIEND_TEST(BufferManagerBackendTest, Test1);
  FRIEND_TEST(BufferManagerBackendTest, Test2);
  FRIEND_TEST(BufferManagerBackendTest, Test3);
  FRIEND_TEST(BufferManagerBackendTest, Test4);
  FRIEND_TEST(BufferManagerBackendTest, Test5);
};

}

#include "PDBBufferManagerBackEndTemplate.cc"

#endif


