

/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/


#ifndef STORAGE_MGR_C
#define STORAGE_MGR_C

#include <fcntl.h>
#include <iostream>
#include <sstream>
#include "PDBStorageManager.h"
#include "PDBPage.h"
#include "PDBStorageFileWriter.h"
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>
#include <utility>
#include <stdio.h>
#include <sys/mman.h>
#include <cstring>

namespace pdb {

size_t PDBStorageManager :: getPageSize () {

	if (!initialized) {
		cerr << "Can't call getPageSize () without initializing the storage manager\n";
		exit (1);
	}

	return pageSize;
}

PDBStorageManager :: ~PDBStorageManager () {

	if (!initialized)
		return;

	// loop through all of the pages currently in existence, and write back each of them
	for (auto &a : allPages) {

		// if the page is not dirty, do not do anything
		PDBPagePtr &me = a.second;
		if (!me->isDirty ())
			continue;

		pair <PDBSetPtr, long> whichPage = make_pair (me->getSet (), me->whichPage ());

		// if we don't know where to write it, figure it out
		if (pageLocations.count (whichPage) == 0) {
			me->getLocation ().startPos = endOfFiles[me->getSet ()];
			pair <PDBSetPtr, size_t> myIndex = make_pair (me->getSet (), me->whichPage ());
			pageLocations[myIndex] = me->getLocation ();
			endOfFiles[me->getSet ()] += (MIN_PAGE_SIZE << me->getLocation ().numBytes);
		}
		
		lseek (fds[me->getSet ()], me->getLocation ().startPos, SEEK_SET);
		write (fds[me->getSet ()], me->getBytes (), MIN_PAGE_SIZE << me->getLocation ().numBytes);
	}

	// and unmap the RAM
	munmap (memBase, pageSize * numPages);

	remove (metaDataFile.c_str ());
	PDBStorageFileWriter myMetaFile (metaDataFile);

	// now, write out the meta-data
	myMetaFile.putUnsignedLong("pageSize", pageSize);
	myMetaFile.putLong("logOfPageSize", logOfPageSize);
	myMetaFile.putUnsignedLong("numPages", numPages);
	myMetaFile.putString ("tempFile", tempFile);
	myMetaFile.putString ("metaDataFile", metaDataFile);
	myMetaFile.putString ("storageLoc", storageLoc);
	
	// get info on each of the sets
	vector <string> setNames;
	vector <string> dbNames;
	vector <string> endPos;
	for (auto &a : endOfFiles) {
		setNames.push_back (a.first->getSetName ());
		dbNames.push_back (a.first->getDBName ());
		endPos.push_back (to_string (a.second));	
	}

	myMetaFile.putStringList ("setNames", setNames);
	myMetaFile.putStringList ("dbNames", dbNames);
	myMetaFile.putStringList ("endPosForEachFile", endPos);

	// now, for each set, write the list of page positions
	map <string, vector <string>> allFiles;
	for (auto &a : pageLocations) {
		string key = a.first.first->getSetName () + "." + a.first.first->getDBName ();
		string value = to_string (a.first.second) + "." + to_string (a.second.startPos) + "." + to_string (a.second.numBytes);
		if (allFiles.count (key) == 0) {
			vector <string> temp;
			temp.push_back (value);
			allFiles[key] = temp;	
		} else {
			allFiles[key].push_back (value);
		}
	}

	// and now write out all of the page positions
	for (auto &a : allFiles) {
		myMetaFile.putStringList (a.first, a.second);
	}
	
	myMetaFile.save ();
}

void PDBStorageManager :: initialize (std :: string metaDataFile) {

	initialized = true;

	// first, get the basic info and initialize everything
	PDBStorageFileWriter myMetaFile (metaDataFile);

	// we use these to grab metadata
	uint64_t utemp;
	int64_t temp;

	myMetaFile.getUnsignedLong("pageSize", utemp);
	pageSize = utemp;
	myMetaFile.getLong("logOfPageSize", temp);
	logOfPageSize = temp;
	myMetaFile.getUnsignedLong("numPages", utemp);
	numPages = utemp;
	myMetaFile.getString ("tempFile", tempFile);
	myMetaFile.getString ("storageLoc", storageLoc);
	initialize (tempFile, pageSize, numPages, metaDataFile, storageLoc);	

	// now, get everything that we need for the end positions
	vector <string> setNames;
	vector <string> dbNames;
	vector <string> endPos;
	
	myMetaFile.getStringList ("setNames", setNames);
	myMetaFile.getStringList ("dbNames", dbNames);
	myMetaFile.getStringList ("endPosForEachFile", endPos);

	// and set up the end positions
	for (int i = 0; i < setNames.size (); i++) {
		PDBSetPtr mySet = make_shared <PDBSet> (setNames[i], dbNames[i]);
		size_t end = stoul (endPos[i]);
		endOfFiles[mySet] = end;
	}

	// now, we need to get the file mappings
	for (auto &a : endOfFiles) {
		vector <string> mappings;
		myMetaFile.getStringList (a.first->getSetName () + "." + a.first->getDBName (), mappings);
		for (auto &b : mappings) {

			// the format of the string in the list is pageNum.startPos.len
			stringstream tokenizer (b); 
			string pageNum, startPos, len;
			getline (tokenizer, pageNum, '.');
			getline (tokenizer, startPos, '.');
			getline (tokenizer, len, '.');
			
			size_t page = stoul (pageNum);
			PDBPageInfo tempInfo;
			tempInfo.startPos = stoul (startPos);
			tempInfo.numBytes = stoul (len);
			pageLocations[make_pair (a.first, page)] = tempInfo;
		}	
	}
}

void PDBStorageManager :: initialize (std :: string tempFileIn, size_t pageSizeIn, size_t numPagesIn,
	std :: string metaFile, std :: string storagLocIn) {	

	initialized = true;
	storageLoc = std::move(storagLocIn);
	pageSize = pageSizeIn;
	tempFile = tempFileIn;
	numPages = numPagesIn;
	metaDataFile = std::move(metaFile);
	tempFileFD = open ((storageLoc + "/" + tempFileIn).c_str (), O_CREAT | O_RDWR, 0666);

	// there are no currently available positions
	logOfPageSize = -1;
	size_t curSize;
	for (curSize = MIN_PAGE_SIZE; curSize <= pageSize; curSize *= 2) {
		vector <size_t> temp;
		vector <void *> tempAgain;
		availablePositions.push_back (temp);	
		emptyMiniPages.push_back (tempAgain);
		logOfPageSize++;
	}

	// but the last used position is zero
	lastTempPos = 0;

	// as is the last time tick
	lastTimeTick = 0;

	if (curSize != pageSize * 2) {
		std :: cerr << "Error: the page size must be a power of two.\n";
		exit (1);
	}

	// now, allocate the RAM
	char *mapped;
        mapped = (char *) mmap (nullptr, pageSize * numPages, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);

        // make sure that it actually worked
        if (mapped == MAP_FAILED) {
                std :: cerr <<  "Could not memory map; error is " << strerror (errno);
                exit (1);
        }
	
	memBase = mapped;

	// and create a bunch of pages
	for (int i = 0; i < numPages; i++) {
		emptyFullPages.push_back (mapped + (pageSize * i));
	}
	
}

void PDBStorageManager :: registerMiniPage (PDBPagePtr registerMe) {

	// first, compute the page this guy is on
	void *whichPage = (char *) memBase + ((((char *) registerMe->getBytes () - (char *) memBase) / pageSize) * pageSize);

	// now, add him to the list of constituent pages
	constituentPages [whichPage].push_back (registerMe);

	// this guy is now pinned
	pinParent (registerMe);
}


void PDBStorageManager :: freeAnonymousPage (PDBPagePtr me) {

	// first, unpin him, if he's not unpinned
	unpin (me);	

	// recycle his location
	PDBPageInfo temp = me->getLocation ();
	if (temp.startPos != -1) {
		availablePositions[temp.numBytes].push_back (temp.startPos); 
	}
	
	// if this guy as no associated memory, get outta here
	if (me->getBytes () == nullptr)
		return;

	// if he is not dirty, it means that he has an associated spot on disk.  Kill it so we can reuse
	if (!me->isDirty ()) {
		availablePositions[me->getLocation ().numBytes].push_back (me->getLocation ().startPos);
	}

	// now, remove him from the set of constituent pages
	void *whichPage = (char *) memBase + ((((char *) me->getBytes () - (char *) memBase) / pageSize) * pageSize);
	for (int i = 0; i < constituentPages[whichPage].size (); i++) {
		if (me == constituentPages[whichPage][i]) {
			constituentPages[whichPage].erase (constituentPages[whichPage].begin () + i);

			// if there are no pinned pages we need to recycle his page
			if (constituentPages[whichPage].empty()) {
				if (numPinned[whichPage] < 0) {
					pair <void *, unsigned> id = make_pair (whichPage, -numPinned[whichPage]);
					lastUsed.erase (id);
				}
				emptyFullPages.push_back (whichPage);	
			}
			
			return;
		}
	}
}

void PDBStorageManager :: downToZeroReferences (PDBPagePtr me) {

	// first, see whether we are actually buffered in RAM
	if (me->getBytes () == nullptr) {

		// we are not, so there is no reason to keep this guy around.  Kill him.
		pair <PDBSetPtr, long> whichPage = make_pair (me->getSet (), me->whichPage ());
		allPages.erase (whichPage);
	} else {
		unpin (me);
	}
}

void PDBStorageManager :: createAdditionalMiniPages(int64_t whichSize) {
	
	// first, we see if there is a page that we can break up; if not, then make one
	if (emptyFullPages.empty()) {
		
		// if there are no pages, give a fatal error
		if (lastUsed.empty()) {
			std :: cerr << "This is really bad.  We seem to have run out of RAM in the storage manager.\n";
			std :: cerr << "I suspect that there are too many pages pinned.\n";
			exit (1);
		}

		// find the LRU
		auto it = lastUsed.begin ();
		auto page = *it;

		// now let all of the contituent pages know the RAM is no longer usable
		for (auto &a: constituentPages[page.first]) {
			if (a->isAnonymous ()) {
				if (a->isDirty ()) {
					if (availablePositions[a->getLocation().numBytes].empty()) {
						a->getLocation ().startPos = lastTempPos;
						lastTempPos += (MIN_PAGE_SIZE << a->getLocation ().numBytes);	
					} else {
						a->getLocation ().startPos = availablePositions[a->getLocation ().numBytes].back ();
						availablePositions[a->getLocation ().numBytes].pop_back ();
					}
					lseek (tempFileFD, a->getLocation ().startPos, SEEK_SET);
					write (tempFileFD, a->getBytes (), MIN_PAGE_SIZE << a->getLocation ().numBytes);
				}
			} else {
				if (a->isDirty ()) {
					PDBPageInfo myInfo = a->getLocation ();
					lseek (fds[a->getSet ()], myInfo.startPos, SEEK_SET);
					write (fds[a->getSet ()], a->getBytes (), MIN_PAGE_SIZE << myInfo.numBytes);
				}	

				// if the number of outstanding references is zero, just kill it
				if (a->numRefs () == 0) {
					pair <PDBSetPtr, long> whichPage = make_pair (a->getSet (), a->whichPage ());
					allPages.erase (whichPage);
				}
			}
			a->setClean ();
			a->setBytes (nullptr);
		}

		// and erase the page	
		constituentPages[page.first].clear ();
		emptyFullPages.push_back (page.first);
		lastUsed.erase (page);
		numPinned.erase (page.first);
	}

	// now, we have a big page, so we can break it up into mini-pages
	size_t inc = MIN_PAGE_SIZE << whichSize;
	for (size_t offset = 0; offset < pageSize; offset += inc) {
		emptyMiniPages[whichSize].push_back (((char *) emptyFullPages.back ()) + offset);
	}

	// set the number of pinned pages to zero... we will always pin this page subsequently,
	// so no need to insert into the LRU queue
	numPinned[emptyFullPages.back ()] = 0;

	// and get rid of the empty full page
	emptyFullPages.pop_back ();	
}


void PDBStorageManager :: freezeSize (PDBPagePtr me, size_t numBytes) {

	if (me->sizeIsFrozen ()) {
		std :: cerr << "You cannot freeze the size of a page twice.\n";
		exit (1);
	}

	me->freezeSize ();

	size_t bytesRequired = 0;
	size_t curSize = MIN_PAGE_SIZE;
	for (; curSize < numBytes; curSize = (curSize << 1)) {
		bytesRequired++;
	}

	me->getLocation ().numBytes = bytesRequired;
}

void PDBStorageManager :: unpin (PDBPagePtr me) {
	
	if (!me->isPinned ()) 
		return;

	// it is no longer pinned
	me->setUnpinned ();

	// freeze the size, if needed
	if (!me->sizeIsFrozen ())
		freezeSize (me, pageSize);

	// first, we find the parent of this guy
	void *memLoc = (char *) memBase + ((((char *) me->getBytes () - (char *) memBase) / pageSize) * pageSize);

	// and decrement the number of pinned minipages
	if (numPinned[memLoc] > 0)
		numPinned[memLoc]--;

	// if the number of pinned minipages is now zero, put into the LRU structure
	if (numPinned[memLoc] == 0) {
		numPinned[memLoc] = -lastTimeTick;
		lastUsed.insert (make_pair (memLoc, lastTimeTick));
		lastTimeTick++;
	}

	// now that the page is unpinned, we find a physical location for it
	pair <PDBSetPtr, long> whichPage = make_pair (me->getSet (), me->whichPage ());
	if (!me->isAnonymous ()) {

		// if we don't know where to write it, figure it out
		if (pageLocations.count (whichPage) == 0) {
			me->getLocation ().startPos = endOfFiles[me->getSet ()];
			pair <PDBSetPtr, size_t> myIndex = make_pair (me->getSet (), me->whichPage ());
			pageLocations[whichPage] = me->getLocation ();
			endOfFiles[me->getSet ()] += (MIN_PAGE_SIZE << me->getLocation ().numBytes);
		}
	} 
}

void PDBStorageManager :: pinParent (PDBPagePtr me) {

	// first, we determine the parent of this guy
	void *whichPage = (char *) memBase + ((((char *) me->getBytes () - (char *) memBase) / pageSize) * pageSize);

	// and increment the number of pinned minipages
	if (numPinned[whichPage] < 0) {
		pair <void *, unsigned> id = make_pair (whichPage, -numPinned[whichPage]);
		lastUsed.erase (id);
		numPinned[whichPage] = 1;
	} else {
		numPinned[whichPage]++;
	}
}

void PDBStorageManager :: repin (PDBPagePtr me) {
	
	// first, we need to see if this page is currently pinned
	if (me->isPinned ()) 
		return;

	// it is not currently pinned, so see if it is in RAM
	if (me->getBytes () != nullptr) {

		// it is currently pinned, so mark the parent as pinned
		me->setPinned ();
		pinParent (me);
		return;
	}

	// it is an anonymous page, so we have to look up its location
	PDBPageInfo myInfo = me->getLocation ();

	// get space for it... first see if the space is available
	if (emptyMiniPages[myInfo.numBytes].empty()) {
		createAdditionalMiniPages (myInfo.numBytes);
	}

	me->setBytes (emptyMiniPages[myInfo.numBytes].back ());
	emptyMiniPages[myInfo.numBytes].pop_back ();

	if (me->isAnonymous ()) {
		lseek (tempFileFD, myInfo.startPos, SEEK_SET);
		read (tempFileFD, me->getBytes (), MIN_PAGE_SIZE << myInfo.numBytes);
	} else {
		lseek (fds[me->getSet ()], myInfo.startPos, SEEK_SET);
		read (fds[me->getSet ()], me->getBytes (), MIN_PAGE_SIZE << myInfo.numBytes);
	}
		
	registerMiniPage (me);
	me->setPinned ();
}

PDBPageHandle PDBStorageManager :: getPage () {
	return getPage (pageSize);
}

PDBPageHandle PDBStorageManager :: getPage (size_t maxBytes) {
		
	if (!initialized) {
		cerr << "Can't call getPageSize () without initializing the storage manager\n";
		exit (1);
	}

	if (maxBytes > pageSize) {
		std :: cerr << maxBytes << " is larger than the system page size of " << pageSize << "\n";
	}

	// figure out the size of the page that we need
	size_t bytesRequired = 0;
	size_t curSize = MIN_PAGE_SIZE;
	for (; curSize < maxBytes; curSize = (curSize << 1))
		bytesRequired++;

	void *space;

	// get space for it... first see if the space is available
	if (emptyMiniPages[bytesRequired].empty()) {
		createAdditionalMiniPages (bytesRequired);
	}

	space = emptyMiniPages[bytesRequired].back ();
	emptyMiniPages[bytesRequired].pop_back ();

	PDBPagePtr returnVal = make_shared <PDBPage> (*this);
	returnVal->setMe (returnVal);
	returnVal->setPinned ();
	returnVal->setDirty ();
	returnVal->setBytes (space);
	returnVal->setAnonymous (true);
	returnVal->getLocation ().numBytes = bytesRequired;
	registerMiniPage (returnVal);
	return make_shared <PDBPageHandleBase> (returnVal);
}

PDBPageHandle PDBStorageManager :: getPage(PDBSetPtr whichSet, uint64_t i) {
		
	if (!initialized) {
		cerr << "Can't call getPageSize () without initializing the storage manager\n";
		exit (1);
	}

	// open the file, if it is not open
	if (fds.count (whichSet) == 0) {
		int fd = open ((storageLoc + "/" + whichSet->getSetName () + "." + whichSet->getDBName ()).c_str (), 
		O_CREAT | O_RDWR, 0666);
		fds[whichSet] = fd;

		if (endOfFiles.count (whichSet) == 0)
			endOfFiles[whichSet] = 0;
	}

	// make sure we don't have a null table
	if (whichSet == nullptr) {
		cerr << "Can't allocate a page with a null table!!\n";
		exit (1);
	}
	
	// next, see if the page is already in existence
	pair <PDBSetPtr, size_t> whichPage = make_pair (whichSet, i);
	if (allPages.count (whichPage) == 0) {

		void *space;
		PDBPagePtr returnVal;

		// it is not there, so see if we have previously created it
		if (pageLocations.count (whichPage) == 0) {

			// we have not previously created it
			PDBPageInfo myInfo;
			myInfo.startPos = i;
			myInfo.numBytes = logOfPageSize;
				
			// get space for it... first see if the space is available
			if (emptyMiniPages[myInfo.numBytes].empty()) {
				createAdditionalMiniPages (myInfo.numBytes);
			}

			space = emptyMiniPages[myInfo.numBytes].back ();
			emptyMiniPages[myInfo.numBytes].pop_back ();
			returnVal = make_shared <PDBPage> (*this);
			returnVal->setMe (returnVal);
			returnVal->setPinned ();
			returnVal->setDirty ();
			returnVal->setSet (whichSet);
			returnVal->setPageNum (i);
			returnVal->setBytes (space);
			returnVal->setAnonymous (false);
			returnVal->getLocation () = myInfo;

		// we have previously created it, so load it up
		} else {

			PDBPageInfo &myInfo = pageLocations[whichPage];

			// get space for it... first see if the space is available
			if (emptyMiniPages[myInfo.numBytes].empty()) {
				createAdditionalMiniPages (myInfo.numBytes);
			}

			space = emptyMiniPages[myInfo.numBytes].back ();
			emptyMiniPages[myInfo.numBytes].pop_back ();

			// read the data from disk
			lseek (fds[whichSet], myInfo.startPos, SEEK_SET);
			read (fds[whichSet], space, MIN_PAGE_SIZE << myInfo.numBytes);
			returnVal = make_shared <PDBPage> (*this);
			returnVal->setMe (returnVal);
			returnVal->setPinned ();
			returnVal->setClean ();
			returnVal->setSet (whichSet);
			returnVal->setPageNum (i);
			returnVal->setBytes (space);
			returnVal->setAnonymous (false);
			returnVal->getLocation () = myInfo;
		}

		registerMiniPage (returnVal);
		allPages [whichPage] = returnVal;
		return make_shared <PDBPageHandleBase> (returnVal);
	}

	// it is there, so return it
	repin (allPages [whichPage]);
	return make_shared <PDBPageHandleBase> (allPages [whichPage]);
}

}

#endif


