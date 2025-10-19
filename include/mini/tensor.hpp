#pragma once
#include "types.hpp"
#include "storage.hpp"
#include "utilities.hpp"
#include <memory>
#include <sstream>

class Tensor{
    public: 
    Storage* storage;
    Shape shape;
    uint8_t ndim;
    Stride stride;
    Offset offset;
    DType dtype;
    bool is_leaf;
    bool requires_grad;
    u_int64_t numel;
    Tensor* grad;



    Tensor(Shape shape, DType dtype = float32, Device device = std::make_pair("cpu",0), bool is_leaf = true, bool requires_grad = false){
        this->shape = shape;
        this->dtype = dtype;
        this->stride = std::vector<int>(shape.size());
        for (int i = 0; i < shape.size(); i++){
            if (i == 0){
                this->stride[i] = 1;
            }
            else{
                this->stride[i] = this->stride[i-1] * shape[i-1];
            }
        }

        this->offset = 0;
        this->ndim = shape.size();
        this->is_leaf = is_leaf;
        this->requires_grad = requires_grad;
        this->numel = 1;
        for (auto val:shape){
            this->numel *= val;
        }
        this->grad = nullptr;

        uint64_t bytes = 1;
        for (auto val:shape){
            bytes *= val;
        }
        bytes *= size_of_dtype(dtype);
        
        this->storage = new Storage(bytes, device, 1);
        
    }


    uint64_t numel(){
        uint64_t numel = 1;
        for (auto val:this->shape){
            numel *= val;
        }
        return numel;

    };






    friend std::ostream& operator <<(std::ostream& os, const Tensor& t){

            os << "Tensor(" << t.shape << ", " << t.dtype << ", " << t.storage->device << ", " << t.is_leaf << ", " << t.requires_grad << ")";



            return os;
    }







};



