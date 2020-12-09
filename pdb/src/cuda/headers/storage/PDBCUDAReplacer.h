#ifndef PDB_CUDA_REPLACER
#define PDB_CUDA_REPLACER

#include <iostream>
#include <PDBCUDAConfig.h>
#include <vector>
#include <set>

namespace pdb {

/**
 * ClockReplacer implements the clock replacement policy, which approximates the Least Recently Used policy.
 */
    class ClockReplacer {
    public:
        /**
        * Create a new ClockReplacer.
        * @param num_pages the maximum number of pages the ClockReplacer will be required to store
        */
        explicit ClockReplacer(size_t num_pages);
        /**
         * Destroys the ClockReplacer.
         */
        ~ClockReplacer();

        bool Victim(frame_id_t* frame_id);

        void Pin(frame_id_t frame_id);

        void Unpin(frame_id_t frame_id);

        size_t Size();

    private:
        void incrementIterator(unsigned int& it);

    private:
        unsigned int totalSize;
        std::vector<ref_bit> buffer;
        std::set<frame_id_t> replacer;
        unsigned int clock_hand;
    };
}
#endif