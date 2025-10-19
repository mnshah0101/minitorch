#pragma once
#include "types.hpp"
#include <memory>


class Storage {
    public:

    void* data;
    Device device;
    u_int64_t nbytes;
    u_int16_t alignment;

        Storage(uint64_t bytes, Device device, u_int16_t alignment){
            this->data = malloc(bytes);
            this->device = device;
            this->nbytes = bytes;
            this->alignment = alignment;
        }

        Storage(){
            this->data = nullptr;
            this->device = std::make_pair("cpu",0);
            this->nbytes = 0;
            this->alignment = 0;
        }

        ~Storage(){
            free(this->data);
        }



};
