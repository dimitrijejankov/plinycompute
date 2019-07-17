

#ifndef FE_STORAGE_MGR_H
#define FE_STORAGE_MGR_H


#include <map>
#include <memory>
#include "PDBBufferManagerInterface.h"
#include "PDBBufferManagerImpl.h"
#include <queue>
#include <set>
#include <PDBBufferManagerImpl.h>
#include <BufGetPageRequest.h>
#include <BufGetAnonymousPageRequest.h>
#include <BufReturnPageRequest.h>
#include <BufReturnAnonPageRequest.h>
#include <BufFreezeSizeRequest.h>
#include <BufPinPageRequest.h>
#include <BufUnpinPageRequest.h>

// this is needed so we can declare friend tests here
#include <gtest/gtest_prod.h>


/**
 * This is the part of the storage manager that is running in the front end.
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
 * The front end is where all of the machinery to make the storage manager actually
 * work is running.
 *
 * The big difference between the front end and the back end storage manager is that
 * the latter simply forwards actions on pages to the front end storage manager to
 * be handled.  For example, if someone calls GetPage () at the back end, the back end
 * creates an object of type GetPageRequest detailing the request, and sends it to
 * the front end.  The front end's handler for that request creates the requested page,
 * and then sends it (via a call to sendPageToBackend ()) back to the back end.
 * Or if the destructor on a PDBPage is called at the backed (meaning that a page
 * received from sendPageToBackend () no longer has any references) then
 * the backed creates an object of type StoReturnPageRequest with information on the page
 * that has no more references, and lets the front end know that all copies (of the
 * copy) of a page sent via sendPageToBackend () are now dead, and that
 * the front end should take appropriate action.
 *
 * Regarding concurrency and thread safety. [IMPORTANT!]
 *
 * The frontend is designed so that it can handle concurrent requests from different pages.
 * Meaning if the backend makes a request to get/return/pin/unpin pages (db1, set1, 0) and (db1, set1, 1)
 * it is perfectly thread safe. But getting and returning the for example (db1, set1, 0) at the same time is NOT!
 * Therefore it is the responsibility of the backend to ensure that requests for the same page are not sent at the
 * same time.
 */
namespace pdb {

class PDBBufferManagerFrontEnd : public PDBBufferManagerImpl {

public:

  // initializes the the storage manager
  explicit PDBBufferManagerFrontEnd(pdb::NodeConfigPtr config) : PDBBufferManagerImpl(std::move(config)) {};

  PDBBufferManagerFrontEnd(std::string tempFileIn, size_t pageSizeIn, size_t numPagesIn, std::string metaFile, std::string storageLocIn);

  ~PDBBufferManagerFrontEnd() override = default;

  // forwards the page to the backend
  bool forwardPage(pdb::PDBPageHandle &page,  PDBCommunicatorPtr &communicator, std::string &error);

  // init
  void init() override;

  // register the handlers
  void registerHandlers(PDBServer &forMe) override;

  // returns the backend
  virtual PDBBufferManagerInterfacePtr getBackEnd();

protected:

  // init the forwarding
  void initForwarding(pdb::PDBPageHandle &page);

  // finish the forwarding
  void finishForwarding(pdb::PDBPageHandle &page);

  // handles the get page request from the backend
  template <class T>
  std::pair<bool, std::string> handleGetPageRequest(pdb::Handle<pdb::BufGetPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the get anonymous page request from the backend
  template <class T>
  std::pair<bool, std::string> handleGetAnonymousPageRequest(pdb::Handle<pdb::BufGetAnonymousPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the return page request from the backend
  template <class T>
  std::pair<bool, std::string> handleReturnPageRequest(pdb::Handle<pdb::BufReturnPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the return anonymous page request
  template <class T>
  std::pair<bool, std::string> handleReturnAnonPageRequest(pdb::Handle<pdb::BufReturnAnonPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the freeze size request from the backend
  template <class T>
  std::pair<bool, std::string> handleFreezeSizeRequest(pdb::Handle<pdb::BufFreezeSizeRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the pin page request from the backend
  template <class T>
  std::pair<bool, std::string> handlePinPageRequest(pdb::Handle<pdb::BufPinPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the unpin page request from the backend
  template <class T>
  std::pair<bool, std::string> handleUnpinPageRequest(pdb::Handle<pdb::BufUnpinPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the logic for the forwarding
  template <class T>
  bool handleForwardPage(pdb::PDBPageHandle &page, std::shared_ptr<T> &communicator, std::string &error);

  // mark the tests for the frontend
  FRIEND_TEST(BufferManagerFrontendTest, Test1);
  FRIEND_TEST(BufferManagerFrontendTest, Test2);
  FRIEND_TEST(BufferManagerFrontendTest, Test3);
  FRIEND_TEST(BufferManagerFrontendTest, Test4);
  FRIEND_TEST(BufferManagerFrontendTest, Test5);
  FRIEND_TEST(BufferManagerFrontendTest, Test6);
  FRIEND_TEST(BufferManagerFrontendTest, Test7);
  FRIEND_TEST(BufferManagerFrontendTest, Test8);

  // sends a page to the backend via the communicator
  template <class T>
  bool sendPageToBackend(PDBPageHandle page, std::shared_ptr<T> &sendUsingMe, std::string &error);

  // Logger to debug information
  PDBLoggerPtr logger;

  // this keeps track of what pages we have sent to the backend
  // in the case that the backend fails this is simply cleared
  // when a page is released by the backend the entry is removed
  map <std::pair<PDBSetPtr, size_t>, PDBPageHandle, PDBPageCompare> sentPages;

  /**
   * All the pages that we are currently forwarding
   */
  set <std::pair<PDBSetPtr, size_t>, PDBPageCompare> forwarding;

  /**
   * The mutex to keep this thing synced
   */
  std::mutex m;

  /**
   * Used to sync page forwarding
   */
  std::condition_variable cv;
};

}

// include the definitions for the storage manager handlers
#include <PDBBufferManagerFrontEndTemplate.cc>

#endif