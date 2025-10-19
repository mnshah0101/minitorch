
#pragma once
#include <vector>
#include <string>
#include <iostream>

enum DType
{

    float16,
    float8,
    float32,
    float64,
    int32,
    int64,
    int8,
    int16
};



using Shape = std::vector<int>;
using Stride = std::vector<int>;
using Offset = u_int64_t;
using Device = std::pair<std::string,int>;

// Operator overloads for output streaming
inline std::ostream& operator<<(std::ostream& os, const DType& dtype) {
    switch(dtype) {
        case float16: os << "float16"; break;
        case float8: os << "float8"; break;
        case float32: os << "float32"; break;
        case float64: os << "float64"; break;
        case int32: os << "int32"; break;
        case int64: os << "int64"; break;
        case int8: os << "int8"; break;
        case int16: os << "int16"; break;
        default: os << "unknown"; break;
    }
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i < shape.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const Device& device) {
    os << device.first << ":" << device.second;
    return os;
}


