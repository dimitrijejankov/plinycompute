

#ifndef FE_STORAGE_MGR_H
#define FE_STORAGE_MGR_H


#include <map>
#include <memory>
#include "PDBStorageManagerInterface.h"
#include "PDBStorageManagerImpl.h"
#include <queue>
#include <set>
#include <PDBStorageManagerImpl.h>
#include <StoGetPageRequest.h>
#include <StoGetAnonymousPageRequest.h>
#include <StoReturnPageRequest.h>
#include <StoReturnAnonPageRequest.h>
#include <StoFreezeSizeRequest.h>
#include <StoPinPageRequest.h>
#include <StoUnpinPageRequest.h>

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
 * and then sends it (via a call to SendPageAccrossConnection ()) back to the back end.
 * Or if the destructor on a PDBPage is called at the backed (meaning that a page
 * received from SendPageAccrossConnection () no longer has any references) then
 * the backed creates an object of type NoMoreReferences with information on the page
 * that has no more references, and lets the front end know that all copies (of the
 * copy) of a page sent via SendPageAccrossConnection () are now dead, and that
 * the front end should take appropriate action.
 */
namespace pdb {

class PDBStorageManagerFrontEnd : public PDBStorageManagerImpl {

public:

  // initializes the the storage manager
  explicit PDBStorageManagerFrontEnd(pdb::NodeConfigPtr config) : PDBStorageManagerImpl(std::move(config)) {};

  PDBStorageManagerFrontEnd(std::string tempFileIn, size_t pageSizeIn, size_t numPagesIn, std::string metaFile, std::string storageLocIn);

  ~PDBStorageManagerFrontEnd() override = default;

  // init
  void init() override;

  // register the handlers
  void registerHandlers(PDBServer &forMe) override;

  // returns the backend
  PDBStorageManagerInterfacePtr getBackEnd();

  // handles the get page request from the backend
  template <class T>
  std::pair<bool, std::string> handleGetPageRequest(pdb::Handle<pdb::StoGetPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the get anonymous page request from the backend
  template <class T>
  std::pair<bool, std::string> handleGetAnonymousPageRequest(pdb::Handle<pdb::StoGetAnonymousPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the return page request from the backend
  template <class T>
  std::pair<bool, std::string> handleReturnPageRequest(pdb::Handle<pdb::StoReturnPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the return anonymous page request
  template <class T>
  std::pair<bool, std::string> handleReturnAnonPageRequest(pdb::Handle<pdb::StoReturnAnonPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the freeze size request from the backend
  template <class T>
  std::pair<bool, std::string> handleFreezeSizeRequest(pdb::Handle<pdb::StoFreezeSizeRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the pin page request from the backend
  template <class T>
  std::pair<bool, std::string> handlePinPageRequest(pdb::Handle<pdb::StoPinPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

  // handles the unpin page request from the backend
  template <class T>
  std::pair<bool, std::string> handleUnpinPageRequest(pdb::Handle<pdb::StoUnpinPageRequest> &request, std::shared_ptr<T> &sendUsingMe);

private:

  // sends a page to the backend via the communicator
  template <class T>
  bool sendPageToBackend(PDBPageHandle page, std::shared_ptr<T> &sendUsingMe, std::string &error);

  // Logger to debug information
  PDBLoggerPtr logger;

  // this keeps track of what pages we have sent to the backend
  // in the case that the backend fails this is simply cleared
  // when a page is released by the backend the entry is removed
  map <std::pair<PDBSetPtr, size_t>, PDBPageHandle, PDBPageCompare> sentPages;

};

}

// include the definitions for the storage manager handlers
#include <PDBStorageManagerFrontEndTemplate.cc>

#endif