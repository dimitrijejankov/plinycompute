
#include "storage/PDBCUDAReplacer.h"

namespace pdb {

    ClockReplacer::ClockReplacer(size_t num_pages) {
        totalSize = num_pages;
        buffer.reserve(num_pages);
        clock_hand = 0;
    }
    ClockReplacer::~ClockReplacer() = default;

    bool ClockReplacer::Victim(frame_id_t *frame_id) {

        while(!replacer.empty()){
            auto iter = replacer.find(clock_hand);
            if(iter != replacer.end()){
                if (buffer[clock_hand]==false){
                    replacer.erase(iter);
                    *frame_id = clock_hand;
                    incrementIterator(clock_hand);
                    return true;
                } else {
                    buffer[clock_hand] = false;
                    incrementIterator(clock_hand);
                }
            } else {
                incrementIterator(clock_hand);
            }
        }
        return false;
    }

    void ClockReplacer::Pin(frame_id_t frame_id) {
        auto iter = replacer.find(frame_id);
        if (iter != replacer.end()){
            replacer.erase(iter);
            buffer[frame_id] = true;
        } else {
            buffer[frame_id] = true;
        }
    }

    void ClockReplacer::Unpin(frame_id_t frame_id) {
        replacer.insert(frame_id);
        buffer[frame_id] = true;
    }

    size_t ClockReplacer::Size() {
        return replacer.size();
    }

    void ClockReplacer::incrementIterator(unsigned int& it){
        if (it == totalSize){
            it = 0;
        } else{
            it++;
        }
    }
}
