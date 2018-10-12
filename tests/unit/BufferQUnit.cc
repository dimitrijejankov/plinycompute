
#ifndef CATALOG_UNIT_H
#define CATALOG_UNIT_H

#include "PDBStorageManager.h"
#include "PDBPageHandle.h"
#include "PDBSet.h"
#include <qunit.h>
#include <cstring>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <vector>

using namespace std;

void writeBytes (int fileName, int pageNum, int pageSize, char *toMe) {

	char foo[1000];
	int num = 0;
	while (num < 900)
		num += sprintf (foo + num, "F: %d, P: %d ", fileName, pageNum);
	memcpy (toMe, foo, pageSize);
	sprintf (toMe + pageSize - 5, "END#");
}

PDBPageHandle createRandomPage (PDBStorageManager &myMgr, vector <PDBSetPtr> &mySets, vector <unsigned> &myEnds) {

	// choose a set
	int whichSet = lrand48 () % mySets.size ();
	
	// choose a length
	size_t len = 16;
	for (; (lrand48 () % 3 != 0) && (len < 64); len *= 2);

	PDBPageHandle returnVal = myMgr.getPage (mySets[whichSet], myEnds[whichSet]);
	writeBytes (whichSet, myEnds[whichSet], len, (char *) returnVal->getBytes ());
	myEnds[whichSet]++;
	returnVal->freezeSize (len);
	return returnVal;
}

static int counter = 0;
PDBPageHandle createRandomTempPage (PDBStorageManager &myMgr) {

	// choose a length
	size_t len = 16;
	for (; (lrand48 () % 3 != 0) && (len < 64); len *= 2);

	PDBPageHandle returnVal = myMgr.getPage ();
	writeBytes (-1, counter, len, (char *) returnVal->getBytes ());
	counter++;
	returnVal->freezeSize (len);
	return returnVal;
}

int main () {

	QUnit::UnitTest qunit(cerr, QUnit::normal);

	// buffer manager and temp page
	cout << "TEST 1...\n";
	{
		PDBStorageManager myMgr;
		myMgr.initialize ("tempDSFSD", 64, 16, "metadata", ".");
		PDBPageHandle page1 = myMgr.getPage();
		PDBPageHandle page2 = myMgr.getPage();
		char *bytes = (char *) page1->getBytes();
		memset(bytes, 'A', 64);
		bytes[63] = 0;
		bytes = (char *) page2->getBytes();
		memset(bytes, 'V', 32);
		bytes[31] = 0;
		page1->unpin ();
		page2->freezeSize (32);
		page2->unpin ();
		for (int i = 0; i < 32; i++) {
			PDBPageHandle page3 = myMgr.getPage();
			PDBPageHandle page4 = myMgr.getPage();
			PDBPageHandle page5 = myMgr.getPage();
		}
		page1->repin ();
		bytes = (char *) page1->getBytes();
		cout << bytes << "\n";
		page2->repin ();
		bytes = (char *) page2->getBytes();
		cout << bytes << "\n";
	}

	cout << "TEST 2...\n";
	{

		PDBStorageManager myMgr;
		myMgr.initialize ("tempDSFSD", 64, 16, "metadata", ".");
		PDBSetPtr set1 = make_shared <PDBSet> ("set1", "DB");
		PDBSetPtr set2 = make_shared <PDBSet> ("set2", "DB");
	        PDBPageHandle page1 = myMgr.getPage(set1, 0);
                PDBPageHandle page2 = myMgr.getPage(set2, 0);
		char *bytes = (char *) page1->getBytes();
		memset(bytes, 'A', 64);
		bytes[63] = 0;
		bytes = (char *) page2->getBytes();
		memset(bytes, 'V', 32);
		bytes[31] = 0;
		page1->unpin ();
		page2->freezeSize (32);
		page2->unpin ();
		for (int i = 0; i < 32; i++) {
			PDBPageHandle page3 = myMgr.getPage(set1, i + 1);
			PDBPageHandle page4 = myMgr.getPage(set1, i + 31);
			PDBPageHandle page5 = myMgr.getPage(set2, i + 1);
			bytes = (char *) page5->getBytes();
			memset(bytes, 'V', 32);
			if (i % 3 == 0) {
				bytes[31] = 0;
				page5->freezeSize (32);
			} else {
				bytes[15] = 0;
				page5->freezeSize (16);
			}
		}
		page1->repin ();
		bytes = (char *) page1->getBytes();
		cout << bytes << "\n";
		page2->repin ();
		bytes = (char *) page2->getBytes();
		cout << bytes << "\n";
	}

	cout << "TEST 3...\n";

	{
		PDBStorageManager myMgr;
		myMgr.initialize ("metadata");
		PDBSetPtr set1 = make_shared <PDBSet> ("set1", "DB");
		PDBSetPtr set2 = make_shared <PDBSet> ("set2", "DB");
	        PDBPageHandle page1 = myMgr.getPage(set1, 0);
                PDBPageHandle page2 = myMgr.getPage(set2, 0);
		char *bytes = (char *) page1->getBytes();
		cout << bytes << "\n";
		bytes = (char *) page2->getBytes();
		cout << bytes << "\n";
	}
		
	cout << "TEST 5...\n";

	{
		PDBStorageManager myMgr;
		myMgr.initialize ("tempDSFSD", 64, 16, "metadata", ".");

		// create the three sets
		vector <PDBSetPtr> mySets;
		vector <unsigned> myEnds;
		for (int i = 0; i < 6; i++) {
			PDBSetPtr set = make_shared <PDBSet> ("set" + to_string (i), "DB");
			mySets.push_back (set);
			myEnds.push_back (0);
		}

		// now, we create a bunch of data and write it to the files, unpinning it
                for (int i = 0; i < 1000; i++) {
                        PDBPageHandle temp = createRandomPage (myMgr, mySets, myEnds);   
			temp->unpin ();
                }

		for (int i = 0; i < 6; i++) {
			std :: cout << "FILE " << i << ":\n";
			for (int j = 0; j < myEnds[i]; j++) {	
				if (j > 100)
					break;
				PDBPageHandle temp = myMgr.getPage (mySets[i], j);
				std :: cout << "\t" << (char *) temp->getBytes () << "\n";
			}
		}

	}
	
	cout << "TEST 6...\n";

	{
		PDBStorageManager myMgr;
		myMgr.initialize ("tempDSFSD", 64, 16, "metadata", ".");

		// create the three sets
		vector <PDBSetPtr> mySets;
		vector <unsigned> myEnds;
		for (int i = 0; i < 6; i++) {
			PDBSetPtr set = make_shared <PDBSet> ("set" + to_string (i), "DB");
			mySets.push_back (set);
			myEnds.push_back (0);
		}

		// first, we create a bunch of data and write it to the files
		vector <PDBPageHandle> myPinnedPages;
		for (int i = 0; i < 10; i++) {
			myPinnedPages.push_back (createRandomPage (myMgr, mySets, myEnds));	
		}
		
		// now, we create a bunch of data and write it to the files, unpinning it
                for (int i = 0; i < 1000; i++) {
                        PDBPageHandle temp = createRandomPage (myMgr, mySets, myEnds);   
			temp->unpin ();
                }

		// now, unpin the temp pages
		for (auto &a : myPinnedPages) {
			a->unpin ();
		}

		// next, we create a bunch of temporary data
		vector <PDBPageHandle> myTempPages;
		for (int i = 0; i < 13; i++) {
			myTempPages.push_back (createRandomTempPage (myMgr));	
		}
		
		// next, we create more data and write it to the files
		// now, we create a bunch of data and write it to the files, unpinning it
                for (int i = 0; i < 1000; i++) {
                        PDBPageHandle temp = createRandomPage (myMgr, mySets, myEnds);   
			temp->unpin ();
                }

		// now, unpin the temporary data
		for (auto &a : myTempPages) {
			a->unpin ();
		}	

		// get rid of the refs to the pinned pages
		myPinnedPages.clear ();

		// now, check the files 
		for (int i = 0; i < 6; i++) {
			std :: cout << "FILE " << i << ":\n";
			for (int j = 0; j < myEnds[i]; j++) {	
				if (j > 100)
					break;
				PDBPageHandle temp = myMgr.getPage (mySets[i], j);
				std :: cout << "\t" << (char *) temp->getBytes () << "\n";
			}
		}

		// and, check the temp pages
		std :: cout << "ANONYMOUS PAGES:\n";
		for (auto &a : myTempPages) {
			a->repin ();
			std :: cout << "\t" << (char *) a->getBytes () << "\n";
		}
		
		myTempPages.clear ();
	}

	cout << "TEST 7...\n";

	{
		PDBStorageManager myMgr;
		myMgr.initialize ("metadata");

		// create the three sets
		vector <PDBSetPtr> mySets;
		for (int i = 0; i < 6; i++) {
			PDBSetPtr set = make_shared <PDBSet> ("set" + to_string (i), "DB");
			mySets.push_back (set);
		}

		for (int i = 0; i < 6; i++) {
			std :: cout << "FILE " << i << ":\n";
			for (int j = 0; j < 100; j++) {	
				PDBPageHandle temp = myMgr.getPage (mySets[i], j);
				std :: cout << "\t" << (char *) temp->getBytes () << "\n";
			}
		}

	}

	QUNIT_IS_TRUE(true);

}

#endif
