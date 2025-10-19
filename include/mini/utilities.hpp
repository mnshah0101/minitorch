#pragma once
#include "types.hpp"




uint64_t size_of_dtype(DType dtype){
    switch(dtype){
        case float16:
            return 2;
        case float8:
            return 1;
        case float32:
            return 4;
        case float64:
            return 8;
        case int32:
            return 4;
        case int64:
            return 8;
        case int8:
            return 1;
        case int16:
            return 2;
    }
};


