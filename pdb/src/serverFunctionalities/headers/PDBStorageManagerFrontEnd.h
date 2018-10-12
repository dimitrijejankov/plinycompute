

#ifndef FE_STORAGE_MGR_H
#define FE_STORAGE_MGR_H


#include <map>
#include <memory>
#include "PDBStorageManagerInterface.h"
#include "PDBStorageManager.h"
#include <queue>
#include <set>
#include <PDBStorageManager.h>

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

class PDBStorageManagerFrontEnd : public PDBStorageManagerInterface {

public:

  PDBStorageManagerFrontEnd() = default;

  ~PDBStorageManagerFrontEnd() override = default;

  // initializes the the storage manager when registered with the server
  // anything that relies on the methods of the @see pdb::ServerFunctionality
  void init() override;

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
  PDBPageHandle getPage(PDBSetPtr whichSet, uint64_t i) override;

  // gets a temporary page that will no longer exist (1) after the buffer manager
  // has been destroyed, or (2) there are no more references to it anywhere in the
  // program.  Typically such a temporary page will be used as buffer memory.
  // since it is just a temp page, it is not associated with any particular
  // set.  On creation, the page is pinned until it is unpinned.
  //
  // Under the hood, this simply finds a mini-page to store the page on (kicking
  // existing data out of the buffer if necessary)
  PDBPageHandle getPage() override;

  // gets a temporary page that is at least minBytes in size
  PDBPageHandle getPage(size_t minBytes) override;

  // gets the page size
  size_t getPageSize() override;

  // register the handlers
  void registerHandlers(PDBServer &forMe) override;

private:

  /**
   * The storage manager
   */
  pdb::PDBStorageManagerPtr storageManager;

};

}

#endif